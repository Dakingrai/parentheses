import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import numpy as np
import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load necessary utilities
from utils.general_utils import MyDataset, load_model
from activation_patching import InterveneOV


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

def save_file(data, file_name):
    """Save data to a JSON file."""
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def is_promoted(max_logit, logit, threshold):
    """
    Returns True if the logit is at least `threshold` times the max logit.
    """
    # round logit, threshold, and max_logit to 4 decimal places
    logit = round(logit, 4)
    threshold = round(threshold, 4)
    max_logit = round(max_logit, 4)
    return logit >= threshold * max_logit

def is_promoted_rank(ranks, threshold_rank=100):
    """
    Returns True if anyone of the rank is smaller threshold rank
    """
    return any(r < threshold_rank for r in ranks)

def get_paren_logit_idx(model_name):
    data_dir = f"data/{model_name}/"
    paren_logit_idx = []
    for n in range(4):
        data_path = f"{data_dir}/train_labeled_last_paren_{n}.json"
        data = read_json(data_path)
        paren_logit_idx.append(data[0]["label_idx"])
    return paren_logit_idx

def get_data(data_path, subtask_paren):
    data = read_json(data_path)

    # Compute n_data
    n_data = len(data)/4
    if len(data) % 4 != 0:
        raise ValueError("Length of data must be divisible by 4")
    
    n_data = int(n_data)
    subtask_n_paren = subtask_paren.count(")")

    if subtask_n_paren == 0:
        raise ValueError("subtask_n_paren is 0")
    
    elif subtask_n_paren == 1:
        new_data = data[:n_data] 
        sample_from_two = random.sample(data[n_data:2*n_data], n_data//3)
        sample_from_three = random.sample(data[2*n_data:3*n_data], n_data//3)
        sample_from_four = random.sample(data[3*n_data:4*n_data], n_data//3)
        precision_data = new_data + sample_from_two + sample_from_three + sample_from_four
        
    elif subtask_n_paren == 2:
        new_data = data[n_data:2*n_data]
        sample_from_one = random.sample(data[:n_data], n_data//3)
        sample_from_three = random.sample(data[2*n_data:3*n_data], n_data//3)
        sample_from_four = random.sample(data[3*n_data:4*n_data], n_data//3)
        precision_data = new_data + sample_from_one + sample_from_three + sample_from_four

    elif subtask_n_paren == 3:
        new_data = data[2*n_data:3*n_data]
        sample_from_one = random.sample(data[:n_data], n_data//3)
        sample_from_two = random.sample(data[n_data:2*n_data], n_data//3)
        sample_from_four = random.sample(data[3*n_data:4*n_data], n_data//3)
        precision_data = new_data + sample_from_one + sample_from_two + sample_from_four
    elif subtask_n_paren == 4:
        new_data = data[3*n_data:4*n_data]
        sample_from_one = random.sample(data[:n_data], n_data//3)
        sample_from_two = random.sample(data[n_data:2*n_data], n_data//3)
        sample_from_three = random.sample(data[2*n_data:3*n_data], n_data//3)
        precision_data =  new_data + sample_from_one + sample_from_two + sample_from_three

    return new_data, precision_data


def calculate_precision_recall_f1(thresholds, precision_data, subtask_n_paren):
    """
    Calculates precision, recall, F1 score, and false positive rate for a given head.
    Parameters:
        thresholds (List[float])
        precision_data (List[Dict]): Contains 'label' and 'paren_logits'
        subtask_n_paren (int): Number of ')' in subtask label
    Returns:
        Tuple of lists: precision, recall, f1_scores, false_positive_rates
    """
    # Initialize metrics
    true_positives = [0] * len(thresholds)
    false_positives = [0] * len(thresholds)
    false_negatives = [0] * len(thresholds)
    true_negatives = [0] * len(thresholds)
    # --- Precision & F1 Calculation (using precision_data) ---
    for entry in precision_data:
        max_logit = entry["paren_logits"]["max-logit"]
        label_n = entry["label"].count(")")
        label_logit = entry["paren_logits"].get(f"{label_n}-paren-logit", 0)
        for i, threshold in enumerate(thresholds):
            # activated = is_promoted(max_logit, label_logit, threshold)
            activated = is_promoted_rank(list(entry["paren_ranks"].values()), threshold_rank=100)
            if activated:
                if label_n == subtask_n_paren:
                    true_positives[i] += 1
                else:
                    false_positives[i] += 1
            elif label_n == subtask_n_paren:
                false_negatives[i] += 1
            else:
                true_negatives[i] += 1
    precision = []
    recall = []
    f1_scores = []
    false_positive_rates = []
    for i in range(len(thresholds)):
        tp, fp, fn, tn = true_positives[i], false_positives[i], false_negatives[i], true_negatives[i]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision.append(prec)
        recall.append(rec)  
        f1_scores.append(f1)
        false_positive_rates.append(fpr)

    return precision, recall, f1_scores, false_positive_rates




def calculate_accuracy_and_confidence(thresholds, data, subtask_paren, paren_token_ids, use_second_highest=False):
    """
    Calculates accuracy and average confidence for correct/incorrect predictions per threshold.

    Confidence = subtask_logit - max(other logits), or second highest if use_second_highest=True.

    Returns:
        Tuple:
            - List of accuracy per threshold
            - List of avg correct confidences per threshold
            - List of avg incorrect confidences per threshold
    """
    subtask_n = subtask_paren.count(")")
    accuracy = [0] * len(thresholds)
    n_total = [0] * len(thresholds)
    correct_confidences = [[] for _ in thresholds]
    incorrect_confidences = [[] for _ in thresholds]

    for entry in data:
        paren_logits = entry["paren_logits"]
        paren_ranks = entry.get("paren_ranks", {})
        max_logit = paren_logits["max-logit"]
        subtask_logit = paren_logits.get(f"{subtask_n}-paren-logit", float("-inf"))
        subtask_rank = paren_ranks.get(f"{subtask_n}-paren-rank", float("inf"))

        # Get ranks of all parens to find the best rank
        all_ranks = [paren_ranks.get(f"{n}-paren-rank", float("inf")) for n in range(1, 5)]
        min_rank = min(all_ranks)

        # Prepare other logits for confidence
        other_logits = [
            paren_logits.get(f"{n}-paren-logit", float("-inf"))
            for n in range(1, 5) if n != subtask_n
        ]
        other_logits = sorted(other_logits, reverse=True)
        ref_logit = (
            other_logits[1] if (use_second_highest and len(other_logits) > 1)
            else other_logits[0] if other_logits else float("-inf")
        )
        confidence = subtask_logit - ref_logit

        for i, threshold in enumerate(thresholds):
            # if is_promoted(max_logit, subtask_logit, threshold):
            n_total[i] += 1
            # if subtask_rank == min_rank:
            if subtask_rank == min_rank and min_rank <= 100:
                accuracy[i] += 1
                correct_confidences[i].append(confidence)
            else:
                incorrect_confidences[i].append(confidence)

    avg_accuracy = [accuracy[i] / n_total[i] if n_total[i] > 0 else 0 for i in range(len(thresholds))]
    avg_correct_conf = [
        sum(c) / len(c) if len(c) > 0 else 0 for c in correct_confidences
    ]
    avg_incorrect_conf = [
        sum(c) / len(c) if len(c) > 0 else 0 for c in incorrect_confidences
    ]

    return avg_accuracy, avg_correct_conf, avg_incorrect_conf


def process_attention_data(proj_results_path, result_path, thresholds, HEADS, subtask_paren, paren_token_ids):
    """
    Processes attention data to compute precision, recall, and F1 scores across different thresholds.
    """
    subtask_n_paren = subtask_paren.count(")")
    activated_heads = {}

    for layer, head in HEADS:
        head_name = f"L{layer}_H{head}"
        activated_heads[head_name] = {}

        data_path = os.path.join(proj_results_path, f"{head_name}_proj.json")
        data, precision_data = get_data(data_path, subtask_paren)

        # Initialize metrics
        precision, recall, f1_scores, false_positive_rates = calculate_precision_recall_f1(thresholds, precision_data, subtask_n_paren)


        accuracy, avg_correct_conf, avg_incorrect_conf = calculate_accuracy_and_confidence(thresholds, data, subtask_paren, paren_token_ids, use_second_highest=True)

        activated_heads[head_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_scores,
            "false_positive_rate": false_positive_rates,
            "accuracy": accuracy,
            "avg_correct_conf": avg_correct_conf,
            "avg_incorrect_conf": avg_incorrect_conf
        }

    result_file = os.path.join(result_path, f"{subtask_n_paren}_results.json")
    save_file(activated_heads, result_file)
    print(f"Results saved to {result_file}")

    del activated_heads, precision, recall, f1_scores, false_positive_rates, accuracy
    clear_cache()
    time.sleep(0.1)


def analyze_macro_f1_results(result_path, paren_token_ids, model, thresholds):
    """
    Computes per-threshold average precision, recall, and F1 for each head (aggregated across subtasks).
    Saves a JSON file mapping head name to its per-threshold scores.
    """

    # Store cumulative values to compute average per threshold
    head_metrics_accumulator = {}

    for paren in paren_token_ids:
        subtask_paren = model.tokenizer.decode(paren)
        subtask_n = subtask_paren.count(")")
        result_file = os.path.join(result_path, f"{subtask_n}_results.json")

        if not os.path.exists(result_file):
            print(f"Warning: Result file {result_file} not found.")
            continue

        all_heads = read_json(result_file)

        for head_name, metrics in all_heads.items():
            if head_name not in head_metrics_accumulator:
                head_metrics_accumulator[head_name] = {
                    "precision": [[] for _ in thresholds],
                    "recall": [[] for _ in thresholds],
                    "f1": [[] for _ in thresholds]
                }

            for i in range(len(thresholds)):
                head_metrics_accumulator[head_name]["precision"][i].append(metrics["precision"][i])
                head_metrics_accumulator[head_name]["recall"][i].append(metrics["recall"][i])
                head_metrics_accumulator[head_name]["f1"][i].append(metrics["f1"][i])

    # Compute average across subtasks for each threshold
    head_avg_metrics = {}
    for head_name, lists in head_metrics_accumulator.items():
        avg_prec = [float(np.mean(p)) for p in lists["precision"]]
        avg_recall = [float(np.mean(r)) for r in lists["recall"]]
        avg_f1 = [float(np.mean(f)) for f in lists["f1"]]

        head_avg_metrics[head_name] = {
            "average_precision": avg_prec,
            "average_recall": avg_recall,
            "macro_f1": avg_f1
        }

    # Save final result
    save_path = os.path.join(result_path, "macro_metrics.json")
    save_file(head_avg_metrics, save_path)
    print(f"\nSaved headwise per-threshold metrics to: {save_path}")

    del head_avg_metrics, head_metrics_accumulator
    clear_cache()
    time.sleep(0.5)


def accuracy_generalization(result_path, paren_token_ids, model, threshold_value=0.8, threshold_index=0):
    """
    Groups heads that exceed a given accuracy threshold (e.g. 0.8) at a specific threshold index
    (e.g. index 0 for threshold=0.5). Saves both generalization groups and per-subtask head sets
    into a single result JSON file.
    """
    from collections import defaultdict

    heads_that_pass = defaultdict(set)  # subtask_id -> set of head names

    # Step 1: Collect passing heads for each subtask
    for paren in paren_token_ids:
        # heads_that_pass[subtask_n] = 
        subtask_paren = model.tokenizer.decode(paren)
        subtask_n = subtask_paren.count(")")
        result_file = os.path.join(result_path, f"{subtask_n}_results.json")

        if not os.path.exists(result_file):
            print(f"Warning: Result file {result_file} not found.")
            continue

        all_heads = read_json(result_file)

        for head_name, metrics in all_heads.items():
            accuracy_list = metrics.get("accuracy", [])
            if accuracy_list[threshold_index] >= threshold_value:
                heads_that_pass[subtask_n].add(head_name)

    # Step 2: Format per-subtask heads
    per_subtask_output = {
        f"{k}-paren": sorted(list(v)) for k, v in heads_that_pass.items()
    }

    # Step 3: Build generalization groups (how many subtasks each head succeeded on)
    head_to_subtasks = defaultdict(set)
    for subtask_id, head_set in heads_that_pass.items():
        for head in head_set:
            head_to_subtasks[head].add(subtask_id)

    generalization_groups = defaultdict(list)
    for head, subtasks in head_to_subtasks.items():
        n = len(subtasks)
        key = f"{n}-general-heads"
        generalization_groups[key].append(head)

    generalization_output = {
        group: {
            "heads": sorted(heads),
            "n_len": len(heads)
        }
        for group, heads in generalization_groups.items()
    }

    # Final result
    final_output = {
        "generalization_groups": generalization_output,
        "per_subtask_heads": per_subtask_output
    }

    save_path = os.path.join(result_path, f"head_generalization_{threshold_index}.json")
    save_file(final_output, save_path)
    print(f"\nSaved full head generalization summary to: {save_path}")

    del final_output, heads_that_pass, per_subtask_output, generalization_groups
    clear_cache()
    time.sleep(0.5)

def process_proj_results(model, folder_name):
    HEADS = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    proj_results_path = f"results/proj_experiment/projections/{folder_name}/final_v/attn/proj"
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9] # threholds for calculating accuracy, precision, recall, f1
    result_path = f"results/proj_experiment/attn_results/{folder_name}/final_v1/"
    create_results_dir(result_path)
    paren_token_ids = get_paren_logit_idx(folder_name)
    for paren in tqdm(paren_token_ids):
        subtask_paren = model.tokenizer.decode(paren)
        process_attention_data(proj_results_path, result_path, thresholds, HEADS, subtask_paren=subtask_paren, paren_token_ids=paren_token_ids)       
    
    # macro F1-score, average precision and recall
    analyze_macro_f1_results(result_path, paren_token_ids, model, thresholds)
    print(f"Finished experiments for {folder_name}")
    # accuracy generalization
    accuracy_generalization(result_path, paren_token_ids, model, threshold_value=0.7)

    del model
    gc.collect()
    torch.cuda.empty_cache()

def count_heads(attn_results_path):
    general_heads = f"{attn_results_path}/head_generalization_0.json"
    general_heads = read_json(general_heads)
    one_general_heads = general_heads["generalization_groups"]["1-general-heads"]["heads"]
    two_general_heads = general_heads["generalization_groups"]["2-general-heads"]["heads"]
    try:
        three_general_heads = general_heads["generalization_groups"]["3-general-heads"]["heads"]
    except:
        three_general_heads = [] 

    try:
        four_general_heads = general_heads["generalization_groups"]["4-general-heads"]["heads"]
    except:
        four_general_heads = []
    
    all_heads = {}
    all_heads["1-general-heads"] = len(one_general_heads)
    all_heads["2-general-heads"] = len(two_general_heads)
    all_heads["3-general-heads"] = len(three_general_heads)
    all_heads["4-general-heads"] = len(four_general_heads)
    return all_heads

def main():
    models = read_json("utils/models.json")
    models = models[-2:]
    # count the number of generalization heads for each model
    model_generalization_heads = {}
    # f1-score
    for m in models:
        pdb.set_trace()
        model_name = m["name"]
        cache_dir = m["cache"]
        model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]

        process_proj_results(model, folder_name)
        # result_dir = f"results/proj_experiment/improve_performance/final_v/{folder_name}/attn/f1-score/last_paren/new"
        attn_results_path = f"results/proj_experiment/attn_results/{folder_name}/final_v1/"

        results = count_heads(attn_results_path)
    
        model_generalization_heads[folder_name] = results
    
    # Step 2: Save the results
    results_path = "results/temp_results/generalization_heads.json"
    create_results_dir("results/temp_results")
    save_file(model_generalization_heads, results_path)


if __name__ == "__main__":
    main()