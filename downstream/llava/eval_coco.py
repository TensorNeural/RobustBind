import os
import json
import math
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import logging
import collections
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


def get_ngrams(sentence, n=4):
    words = sentence.lower().split()
    return [tuple(words[i:i + k]) for k in range(1, n + 1) for i in range(len(words) - k + 1)]


def compute_cider(predictions, references, n=4):
    doc_freq = collections.defaultdict(int)
    ref_len = len(references)
    for refs in references:
        unique_ngrams = set()
        for ref in refs:
            unique_ngrams.update(get_ngrams(ref, n))
        for ng in unique_ngrams:
            doc_freq[ng] += 1

    def tf_idf_vector(sentence):
        tf = collections.Counter(get_ngrams(sentence, n))
        vec = {}
        for ng, cnt in tf.items():
            df = doc_freq.get(ng, 1)
            idf = math.log(max(1.0, ref_len) / df)
            vec[ng] = cnt * idf
        norm = math.sqrt(sum(v ** 2 for v in vec.values()))
        return vec, norm

    scores = []
    for pred, refs in zip(predictions, references):
        vec_hyp, norm_hyp = tf_idf_vector(pred)
        sim_total = 0.0
        for ref in refs:
            vec_ref, norm_ref = tf_idf_vector(ref)
            dot = sum(vec_hyp[k] * vec_ref.get(k, 0.0) for k in vec_hyp)
            if norm_hyp > 0 and norm_ref > 0:
                sim_total += dot / (norm_hyp * norm_ref)
        scores.append(10.0 * sim_total / len(refs))
    return sum(scores) / len(scores)


@torch.inference_mode()
def generate_batch(logger, model, tokenizer, image_processor, image_paths, prompts, device):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_tensors = [image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0] for img in images]
    image_tensor_batch = torch.stack(image_tensors).to(device, dtype=model.get_vision_tower().dtype)

    input_ids_batch = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for p in prompts]
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
        max_new_tokens=32,
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
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--use_unibind", action='store_true', default=True, help="Use Unibind for encoder")
    args = parser.parse_args()

    batch_size = args.batch_size

    torch.set_num_threads(os.cpu_count())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logger(output_path, rank)

    logger.info("Starting COCO Caption Evaluation")
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

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.to(device)
    logger.info(f"model.get_vision_tower() dtype: {model.get_vision_tower().dtype}")

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
        prompts = []
        for _ in batch:
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], "<image>\nDescribe the image.")
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        captions = generate_batch(logger, model, tokenizer, image_processor, image_paths, prompts, device)

        for item, caption in zip(batch, captions):
            gt = item.get("captions", [item.get("caption", "")])
            logger.info(f"Image: {item['image']}")
            logger.info(f"GT: {gt}")
            logger.info(f"Pred: {caption}")

            results.append({
                "image_id": item["image_id"],
                "caption": caption,
                "ground_truth": gt
            })

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, results)

    if rank == 0:
        all_results = [r for sublist in gathered for r in sublist]
        cider = compute_cider(
            [r["caption"] for r in all_results],
            [r["ground_truth"] for r in all_results]
        )

        output_json = os.path.join(output_path, "coco_results.json")
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"✅ Saved {len(all_results)} COCO results to {output_json}")
        logger.info(f"📊 Final COCO CIDEr Score: {cider:.3f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
