import json
import argparse
import random
import os
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def sample_data(data, sample_size, uniform_label):
    if uniform_label:
        label_dict = defaultdict(list)
        for entry in data:
            label_dict[entry['label']].append(entry)
        
        sampled_data = []
        labels = list(label_dict.keys())
        per_label = max(1, sample_size // len(labels))
        
        for label in labels:
            sampled_data.extend(random.sample(label_dict[label], min(per_label, len(label_dict[label]))))
        
        # Adjust final sample size in case of rounding errors
        sampled_data = random.sample(sampled_data, min(len(sampled_data), sample_size))
    else:
        sampled_data = random.sample(data, min(sample_size, len(data)))
    
    return sampled_data

def main():
    parser = argparse.ArgumentParser(description="Randomly sample entries from a JSON file.")
    parser.add_argument("--input_file", type=str, default="./ESC-50/val_data.json", help="Path to the JSON file.")
    parser.add_argument("--output_dir", type=str, default="./ESC-50", help="Directory to save output files.")
    parser.add_argument("--sample_sizes", type=int, nargs='+', default=[500, 1000, 3000, 5000, 8000], help="List of sample sizes.")
    parser.add_argument("--uniform_label", action="store_true", default=False, help="Enable equal sampling from each label.")
    
    args = parser.parse_args()
    data = load_json(args.input_file)
    
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    for sample_size in args.sample_sizes:
        sampled_data = sample_data(data, sample_size, args.uniform_label)
        suffix = "_uniform" if args.uniform_label else ""
        output_file = os.path.join(args.output_dir, f"{base_name}_{sample_size}{suffix}.json")
        save_json(sampled_data, output_file)
        print(f"Saved sampled data to {output_file}")

if __name__ == "__main__":
    main()
