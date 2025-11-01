import os
import json
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from torchvision import transforms
import logging

from downstream.llava.model.builder import load_pretrained_model
from downstream.llava.utils import disable_torch_init
from downstream.llava.constants import IMAGE_TOKEN_INDEX
from downstream.llava.conversation import conv_templates
from downstream.llava.mm_utils import tokenizer_image_token
from perf.profiling import ProfileModelMemory


class RelativePathFormatter(logging.Formatter):
    def __init__(self, rank, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.rank = rank

    def format(self, record):
        run_dir = os.getcwd()
        record.rank = self.rank
        record.relativepath = os.path.relpath(os.path.abspath(record.pathname), run_dir)
        return super().format(record)


def setup_logger(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"eval_rank{rank}.log")
    formatter = RelativePathFormatter(
        rank,
        fmt='[RANK %(rank)d] %(asctime)s - %(relativepath)s:%(lineno)d - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers = [ch, fh]
    return logger


def normalize_answer(s):
    import re, string
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punctuation(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    text = lower(s)
    text = white_space_fix(remove_articles(remove_punctuation(text)))
    return "yes" if text.startswith("yes") else "no" if text.startswith("no") else text


def vqa_soft_accuracy(prediction, answers):
    prediction = normalize_answer(prediction)
    answers = [normalize_answer(ans) for ans in answers]
    return min(1.0, answers.count(prediction) / 3.0)


def ddp_scatter(data, rank, world_size):
    chunk_size = len(data) // world_size
    remainder = len(data) % world_size
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)
    return data[start:end]


@torch.inference_mode()
def generate_batch(logger, model, tokenizer, image_processor, image_paths, prompts, device, use_random_image=False):
    if use_random_image:
        sample_img = image_processor.preprocess(Image.new("RGB", (512, 512)), return_tensors="pt")["pixel_values"][0]
        img_shape = sample_img.shape
        image_tensor_batch = torch.randn((len(prompts), *img_shape), device=device, dtype=model.get_vision_tower().dtype)
    else:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        image_tensors = [image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0] for img in images]
        image_tensor_batch = torch.stack(image_tensors).to(device, dtype=model.get_vision_tower().dtype)

    input_ids_batch = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for p in prompts]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids_batch != tokenizer.pad_token_id).long().to(device)
    input_ids_batch = input_ids_batch.to(device)

    with ProfileModelMemory(model, logger):
        output_ids = model.generate(
            inputs=input_ids_batch,
            attention_mask=attention_mask,
            images=image_tensor_batch,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=8,
            pad_token_id=tokenizer.pad_token_id
        )
    torch.cuda.empty_cache()
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--projector_weight", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--use_unibind", action='store_true', default=False, help="Use Unibind for encoder")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", device_id=rank)
    device = torch.device("cuda", rank)
    world_size = dist.get_world_size()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    root_dir = os.path.join(args.output_dir, timestamp)

    # model_tags = ["unibind", "robustbind2", "robustbind4"]
    model_tags = ["unibind"]
    settings = [
        {
            "name": "clean",
            "val_json_template": "datasets/VQA2/val_data.json",
            "use_random_image": False
        },
        # {
        #     "name": "random",
        #     "val_json_template": "datasets/VQA2/val_data_filtered.json",
        #     "use_random_image": True
        # },
        # {
        #     "name": "eps2",
        #     "val_json_template": "datasets/VQA2/val_data_adv_eps2_{model_tag}.json",
        #     "use_random_image": False
        # },
        # {
        #     "name": "eps4",
        #     "val_json_template": "datasets/VQA2/val_data_adv_eps4_{model_tag}.json",
        #     "use_random_image": False
        # },
    ]

    lora_weights_map = {
        "unibind": None,
        "robustbind2": "./ckpts/vision_eps2_lora_weights.pt",
        "robustbind4": "./ckpts/vision_eps4_lora_weights.pt",
    }

    for setting in settings:
        setting_name = setting["name"]
        val_json_template = setting["val_json_template"]
        use_random = setting["use_random_image"]

        for model_tag in model_tags:
            if setting_name == "random" and model_tag != "unibind":
                continue

            val_json = val_json_template.format(model_tag=model_tag)
            model_out_dir = os.path.join(root_dir, setting_name, model_tag)
            os.makedirs(model_out_dir, exist_ok=True)
            logger = setup_logger(model_out_dir, rank)
            logger.info(f"CLI Arguments: {json.dumps(vars(args), indent=2)}")
            logger.info(f"🧪 Evaluating {model_tag.upper()} [{setting_name}]")

            with open(val_json) as f:
                data = json.load(f)
            if args.max_samples:
                data = data[:args.max_samples]
                
            data = ddp_scatter(data, rank, world_size)
            logger.info(f"📊 Rank {rank} processing {len(data)} samples...")

            disable_torch_init()
            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path=args.model_path,
                model_name=args.model_path,
                model_base=None,
                torch_dtype=torch.float16,
                device=device,
                device_map=None,
                use_unibind=args.use_unibind,
                unibind_pretrain_weights="./ckpts/pretrained_weights_flash_atten_image_patchs.pt",
                projector_weights_path=args.projector_weight,
                unibind_use_lora=lora_weights_map[model_tag] is not None,
                unibind_lora_weights=lora_weights_map[model_tag],
                freeze_projector=True,
                freeze_unibind=True,
                unibind_lora_rank=4,
                unibind_lora_alpha=8
            )
            model = model.to(device)

            results = []
            for i in tqdm(range(0, len(data), args.batch_size), disable=(rank != 0)):
                batch = data[i:i + args.batch_size]
                image_paths = [os.path.join(args.image_root, item["image"]) for item in batch]
                questions = [item["question"] for item in batch]
                answers = [item.get("answers", []) for item in batch]

                prompts = []
                for q in questions:
                    conv = conv_templates["llava_v1"].copy()
                    conv.append_message(conv.roles[0], f"<image>\nQuestion: {q}\nAnswer:")
                    conv.append_message(conv.roles[1], None)
                    prompts.append(conv.get_prompt())

                preds = [normalize_answer(p) for p in generate_batch(
                    logger, model, tokenizer, image_processor,
                    image_paths, prompts, device,
                    use_random_image=use_random
                )]
                accs = [vqa_soft_accuracy(p, a) for p, a in zip(preds, answers)]

                for item, pred, acc in zip(batch, preds, accs):
                    results.append({
                        "image_id": item["image_id"],
                        "question_id": item["question_id"],
                        "question": item["question"],
                        "predicted_answer": pred,
                        "vqa_soft_accuracy": acc
                    })

            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, results)

            if rank == 0:
                all_results = [r for sublist in gathered for r in sublist]
                acc_mean = sum(r["vqa_soft_accuracy"] for r in all_results) / len(all_results)

                output_json = os.path.join(model_out_dir, "vqa_results.json")
                with open(output_json, "w") as f:
                    json.dump(all_results, f, indent=2)

                logger.info(f"✅ Saved {len(all_results)} VQA results to {output_json}")
                logger.info(f"📊 Final VQA Soft Accuracy: {acc_mean:.3f}")

                epsilon_map = {"clean": "None", "random": "None", "eps2": "2/255", "eps4": "4/255"}
                print("\n=== CSV RESULT ===")
                print("Model,Setting,Epsilon,Accuracy")
                print(f"{model_tag},{setting_name},{epsilon_map[setting_name]},{acc_mean:.2f}")

            torch.distributed.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
