#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import tqdm
from collections import Counter

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def generate_vqa_metadata(vqa_root, split, output_dir):
    split_map = {"train": "train2014", "val": "val2014"}
    short = split_map[split]

    questions_file = os.path.join(vqa_root, f"v2_OpenEnded_mscoco_{short}_questions.json")
    annotations_file = os.path.join(vqa_root, f"v2_mscoco_{short}_annotations.json")
    out_path = os.path.join(output_dir, "train_data.json" if split == "train" else "val_data.json")

    if not os.path.exists(questions_file) or not os.path.exists(annotations_file):
        raise FileNotFoundError("Missing VQA questions or annotations.")

    questions = load_json(questions_file)["questions"]
    annotations = load_json(annotations_file)["annotations"]
    ann_map = {ann["question_id"]: ann for ann in annotations}

    entries = []
    for q in tqdm(questions, desc=f"Generating VQA {split} metadata"):
        qid = q["question_id"]
        ann = ann_map[qid]
        answers = [a["answer"] for a in ann["answers"]]
        confidence_hist = dict(Counter(a["answer_confidence"] for a in ann["answers"]))
        entries.append({
            "image": os.path.join(split, f"COCO_{short}_{q['image_id']:012d}.jpg"),
            "image_id": q["image_id"],
            "question": q["question"],
            "question_id": qid,
            "answers": answers,
            "answer_confidence_hist": confidence_hist,
            "multiple_choice_answer": ann["multiple_choice_answer"],
            "question_type": ann["question_type"],
            "answer_type": ann["answer_type"]
        })

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"[✓] VQA-{split} metadata → {out_path} ({len(entries)} entries)")

def main():
    parser = argparse.ArgumentParser(description="Generate VQA2 Open-Ended metadata for LLaVA / OF / VQA scoring and analysis")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to VQA2 directory")
    args = parser.parse_args()

    generate_vqa_metadata(args.dataset_root, "train", "./VQA2")
    generate_vqa_metadata(args.dataset_root, "val", "./VQA2")

if __name__ == "__main__":
    main()
