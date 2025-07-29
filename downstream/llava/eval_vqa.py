import os
import json
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import logging
from datetime import datetime

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


@torch.inference_mode()
def generate_batch(logger, model, tokenizer, image_processor, image_paths, prompts, device):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_tensors = [image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0] for img in images]
    image_tensor_batch = torch.stack(image_tensors).to(device, dtype=model.get_vision_tower().dtype)

    input_ids_batch = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for p in prompts]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids_batch != tokenizer.pad_token_id).long().to(device)
    input_ids_batch = input_ids_batch.to(device)

    # with ProfileModelMemory(model, logger):
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


def ddp_scatter(data, rank, world_size):
    chunk_size = len(data) // world_size
    remainder = len(data) % world_size
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)
    return data[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--projector_weight", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--use_unibind", action='store_true', default=True, help="Use Unibind for encoder")
    args = parser.parse_args()

    batch_size = args.batch_size

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logger(output_path, rank)

    logger.info("Starting VQA Evaluation")
    logger.info(f"Model Dir: {args.model_dir}")
    logger.info(f"Val JSON: {args.val_json}")
    logger.info(f"Image Root: {args.image_root}")
    logger.info(f"Output Dir: {output_path}")
    logger.info(f"Max Samples: {args.max_samples}")
    logger.info(f"Using {world_size} GPUs | Batch size: {batch_size}")

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_dir,
        model_name=os.path.basename(args.model_dir),
        model_base=None,
        torch_dtype=torch.float16,
        device=device,
        device_map=None,
        use_unibind=args.use_unibind,
        unibind_pretrain_weights="./ckpts/pretrained_weights_flash_atten.pt",
        projector_weights_path=args.projector_weight,
        freeze_projector=True,
        freeze_unibind=True,
        unibind_lora_rank=4,
        unibind_lora_alpha=8
    )

    logger.info(f"model.get_vision_tower() dtype: {model.get_vision_tower().dtype}")
    model = model.to(device)

    with open(args.val_json) as f:
        data = json.load(f)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    data = ddp_scatter(data, rank, world_size)
    logger.info(f"📊 Rank {rank} processing {len(data)} samples...")

    results = []
    for i in tqdm(range(0, len(data), batch_size), disable=(rank != 0)):
        batch = data[i:i+batch_size]

        image_paths = [os.path.join(args.image_root, item["image"]) for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item.get("answers", []) for item in batch]

        prompts = []
        for q in questions:
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], f"<image>\nQuestion: {q}\nAnswer:")
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        preds = [normalize_answer(p) for p in generate_batch(logger, model, tokenizer, image_processor, image_paths, prompts, device)]
        accs = [vqa_soft_accuracy(p, a) for p, a in zip(preds, answers)]

        for item, pred, acc in zip(batch, preds, accs):
            logger.info(f"Image: {item['image']}")
            logger.info(f"Q: {item['question']}")
            logger.info(f"A(pred): {pred}")
            logger.info(f"A(gt): {item.get('answers', [])}")
            logger.info(f"Acc: {acc:.2f}")

            results.append({
                "image_id": item["image_id"],
                "question_id": item["question_id"],
                "question": item["question"],
                "predicted_answer": pred,
                "vqa_soft_accuracy": acc
            })

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, results)

    if rank == 0:
        all_results = [r for sublist in gathered for r in sublist]
        acc_mean = sum(r["vqa_soft_accuracy"] for r in all_results) / len(all_results)

        output_json = os.path.join(output_path, "vqa_results.json")
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"✅ Saved {len(all_results)} VQA results to {output_json}")
        logger.info(f"📊 Final VQA Soft Accuracy: {acc_mean:.3f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
