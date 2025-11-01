import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download

from downstream.llava.model.builder import load_pretrained_model
from downstream.llava.utils import disable_torch_init
from downstream.llava.constants import IMAGE_TOKEN_INDEX
from downstream.llava.conversation import conv_templates
from downstream.llava.mm_utils import tokenizer_image_token
import traceback

import re
import string

def normalize_answer(s):
    """
    Normalize free-form VQA answers:
    - Lowercase
    - Strip articles/punctuation/whitespace
    - Clip long 'yes'/'no' answers to just 'yes' or 'no'
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    text = lower(s)
    text = white_space_fix(remove_articles(remove_punctuation(text)))

    if text.startswith('yes'):
        return 'yes'
    if text.startswith('no'):
        return 'no'

    return text

def vqa_soft_accuracy(prediction, answers):
    """
    Compute VQA-style soft accuracy.
    Args:
        prediction (str): model output
        answers (List[str]): 10 annotator answers
    Returns:
        float: accuracy ∈ [0, 1]
    """
    prediction = normalize_answer(prediction)
    answers = [normalize_answer(ans) for ans in answers]
    return min(1.0, answers.count(prediction) / 3.0)

def download_weights(model_path: str, local_cache_dir: str):
    local_model_dir = os.path.join(local_cache_dir, model_path.replace("/", "--"))
    snapshot_download(
        repo_id=model_path,
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    return local_model_dir

def load_model(model_dir: str, torch_dtype=torch.float16):
    disable_torch_init()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_dir,
        model_name=os.path.basename(model_dir),
        model_base=None,
        torch_dtype=torch_dtype,
        device="cuda",
        device_map="auto",
        use_unibind=True,
        unibind_pretrain_weights="./ckpts/pretrained_weights_flash_atten_image_patchs.pt",
        unibind_use_lora=False,
        unibind_lora_rank=4,
        unibind_lora_alpha=8.0,
        unibind_lora_weights=None,
        projector_weights_path=None,
        freeze_projector=True,
        freeze_unibind=True,
    )
    return model.eval(), image_processor, tokenizer

def run_vqa_single(image_path: str, question: str, model, tokenizer, image_processor):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to("cuda")
    image_tensor = image_tensor.to(model.get_vision_tower().dtype)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"""Answer strictly based on the image with yes/no or in 1 or 2 words. Answer in lowercase.
<image>
Question: {question}  
Answer:""")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=8,
            use_cache=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def main():
    # === Config ===
    model_repo = "liuhaotian/llava-v1.6-mistral-7b"
    local_cache = os.path.join(os.getcwd(), ".cache")
    metadata_path = "datasets/VQA2/val_data.json"
    image_root = "/data/datasets/VQA2"
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vqa2_llava_results.json")
    max_samples = 5000

    # === Load model
    model_dir = download_weights(model_repo, local_cache)
    model, image_processor, tokenizer = load_model(model_dir, torch_dtype=torch.float16)

    # === Load data
    with open(metadata_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} VQA2 entries. Running on first {max_samples}...")

    # === Evaluation tracking
    results = []
    vqa_acc_sum = 0.0
    vqa_acc_count = 0

    for i, item in enumerate(tqdm(data[:max_samples])):
        try:
            image_path = os.path.join(image_root, item["image"])
            question = item["question"]
            pred = normalize_answer(run_vqa_single(image_path, question, model, tokenizer, image_processor))

            gt_answers = item.get("answers", [])
            vqa_acc = vqa_soft_accuracy(pred, gt_answers) if gt_answers else 0.0
            vqa_acc_sum += vqa_acc
            vqa_acc_count += 1

            results.append({
                "image_id": item["image_id"],
                "question_id": item["question_id"],
                "question": question,
                "predicted_answer": pred,
                "multiple_choice_answer": item.get("multiple_choice_answer"),
                "ground_truth_answers": gt_answers,
                "vqa_soft_accuracy": vqa_acc
            })

        except Exception as e:
            print(f"[{i}] ❌ Failed on {item['image']}: {e}")
            traceback.print_exc()
            return

    # === Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved {len(results)} answers to {output_path}")

    # === Report final VQA accuracy
    if vqa_acc_count > 0:
        mean_vqa_score = vqa_acc_sum / vqa_acc_count
        print(f"\n📊 Final VQA Soft Accuracy over {vqa_acc_count} questions: {mean_vqa_score:.3f}")
    else:
        print("\n⚠️ No valid VQA accuracy scores computed.")

if __name__ == "__main__":
    main()
