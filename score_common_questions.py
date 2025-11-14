import os
import json
import argparse
from datetime import datetime
import csv
from collections import defaultdict


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_eps_tag(path):
    for p in path.split(os.sep):
        if p.startswith("eps"):
            return p
    return "eps0"


def get_model_tag(path):
    return os.path.basename(os.path.dirname(path))


def get_unchanged_question_ids(clean_json, random_json):
    clean = load_json(clean_json)
    rand = load_json(random_json)

    clean_map = {item['question_id']: item['predicted_answer'].strip().lower() for item in clean}
    rand_map = {item['question_id']: item['predicted_answer'].strip().lower() for item in rand}

    unchanged = {qid for qid in clean_map if qid in rand_map and clean_map[qid] == rand_map[qid]}
    print(f"[✓] Found {len(unchanged)} unchanged question_ids")
    return unchanged


def filter_and_group_results(json_paths, unchanged_qids, output_dir, timestamp):
    result_data = {}
    qid_sets = []

    for path in json_paths:
        model_tag = get_model_tag(path)
        eps_tag = get_eps_tag(path)
        data = load_json(path)
        filtered = [x for x in data if x['question_id'] not in unchanged_qids]
        filtered_map = {x['question_id']: x for x in filtered}
        result_data[(eps_tag, model_tag)] = filtered_map
        qid_sets.append(set(filtered_map.keys()))

        out_path = os.path.join(output_dir, f"vqa_results_{eps_tag}_{model_tag}_filtered_{timestamp}.json")
        save_json(filtered, out_path)
        print(f"[✓] Saved filtered {eps_tag}/{model_tag}: {len(filtered)} entries to {out_path}")

    common_qids = set.intersection(*qid_sets)
    print(f"\n[✓] Found {len(common_qids)} common question_ids across all filtered results.\n")
    return result_data, common_qids


def compute_score_on_common(qids, result_map):
    accs = [float(entry['vqa_soft_accuracy']) for qid, entry in result_map.items() if qid in qids]
    return sum(accs) / len(accs) if accs else 0.0


def print_scores(result_data, common_qids, output_dir, timestamp):
    eps_grouped = defaultdict(dict)
    for (eps_tag, model_tag), result_map in result_data.items():
        eps_grouped[eps_tag][model_tag] = result_map

    for eps_tag in sorted(eps_grouped):
        model_dict = eps_grouped[eps_tag]
        print(f"=== SCORES ON COMMON QUESTIONS — {eps_tag.upper()} ===")
        print("{:<20} {:>10} {:>10}".format("Model", "#Questions", "Avg Acc"))
        print("=" * 42)

        csv_path = os.path.join(output_dir, f"common_scores_{eps_tag}_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "num_questions", "avg_accuracy"])

            for model_tag, result_map in model_dict.items():
                acc = compute_score_on_common(common_qids, result_map)
                print(f"{model_tag:<20} {len(common_qids):>10} {acc:>10.4f}")
                writer.writerow([model_tag, len(common_qids), f"{acc:.4f}"])

        print(f"[✓] Saved CSV to {csv_path}\n")


def analyze_disagreements(result_data, common_qids):
    eps_grouped = defaultdict(dict)
    for (eps_tag, model_tag), result_map in result_data.items():
        eps_grouped[eps_tag][model_tag] = result_map

    for eps_tag in sorted(eps_grouped):
        if "unibind" not in eps_grouped[eps_tag]:
            continue
        base = eps_grouped[eps_tag]["unibind"]

        for variant in ["robustbind2", "robustbind4"]:
            if variant not in eps_grouped[eps_tag]:
                continue
            rb = eps_grouped[eps_tag][variant]

            both_correct = 0
            only_robust = 0
            only_unibind = 0

            for qid in common_qids:
                if qid not in base or qid not in rb:
                    continue
                base_acc = float(base[qid]['vqa_soft_accuracy']) >= 0.5
                rb_acc = float(rb[qid]['vqa_soft_accuracy']) >= 0.5
                if base_acc and rb_acc:
                    both_correct += 1
                elif rb_acc and not base_acc:
                    only_robust += 1
                elif base_acc and not rb_acc:
                    only_unibind += 1

            print(f"=== AGREEMENT ANALYSIS — {eps_tag.upper()} | {variant} ===")
            print(f"  Both correct:         {both_correct}")
            print(f"  Only robustbind:      {only_robust}")
            print(f"  Only unibind:         {only_unibind}")
            print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_json', type=str, required=False,
        default='/data/output/llava/eval/vqa/2025-07-31_10-24/clean/unibind/vqa_results.json')
    parser.add_argument('--random_json', type=str, required=False,
        default='/data/output/llava/eval/vqa/2025-07-31_10-24/random/unibind/vqa_results.json')
    parser.add_argument('--result_jsons', type=str, nargs='+', required=False,
        default=[
            '/data/output/llava/eval/vqa/2025-07-31_10-24/clean/unibind/vqa_results.json',
            '/data/output/llava/eval/vqa/2025-07-31_10-24/clean/robustbind2/vqa_results.json',
            '/data/output/llava/eval/vqa/2025-07-31_10-24/clean/robustbind4/vqa_results.json',
            '/data/output/llava/eval/vqa/2025-07-31_09-08/eps2/unibind/vqa_results.json',
            '/data/output/llava/eval/vqa/2025-07-31_09-08/eps2/robustbind2/vqa_results.json',
            '/data/output/llava/eval/vqa/2025-07-31_09-08/eps2/robustbind4/vqa_results.json',
            '/data/output/llava/eval/vqa/2025-07-31_09-08/eps4/unibind/vqa_results.json',
            '/data/output/llava/eval/vqa/2025-07-31_09-08/eps4/robustbind2/vqa_results.json',
            '/data/output/llava/eval/vqa/2025-07-31_09-08/eps4/robustbind4/vqa_results.json'
        ])
    parser.add_argument('--output_dir', type=str, default='/data/output/llava/eval/vqa/filtered')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    unchanged_qids = get_unchanged_question_ids(args.clean_json, args.random_json)

    result_data, common_qids = filter_and_group_results(
        args.result_jsons, unchanged_qids, args.output_dir, timestamp
    )

    print_scores(result_data, common_qids, args.output_dir, timestamp)
    analyze_disagreements(result_data, common_qids)


if __name__ == "__main__":
    main()
