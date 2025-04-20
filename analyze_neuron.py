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
from activation_patching import InterveneOV, InterveneNeurons


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
            activated = is_promoted(max_logit, label_logit, threshold)
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


def calculate_accuracy(thresholds, data, subtask_paren, paren_token_ids):
    """
    Calculates accuracy across thresholds where:
    - The subtask paren is activated (using is_promoted).
    - The subtask paren has the highest rank (i.e., lowest numerical rank) among all paren_token_ids.

    Parameters:
        thresholds (List[float])
        data (List[Dict]): Each entry contains paren_logits, paren_ranks, etc.
        subtask_paren (str): The string with parentheses (e.g., ')))')
        paren_token_ids (List[int]): List of 1-paren to 4-paren token ids

    Returns:
        List[float]: Accuracy at each threshold
    """
    subtask_n = subtask_paren.count(")")
    accuracy = [0] * len(thresholds)
    n_total = [0] * len(thresholds)

    for entry in data:
        paren_logits = entry["paren_logits"]
        paren_ranks = entry.get("paren_ranks", {})
        max_logit = paren_logits["max-logit"]
        subtask_logit = paren_logits.get(f"{subtask_n}-paren-logit", float("-inf"))
        subtask_rank = paren_ranks.get(f"{subtask_n}-paren-rank", float("inf"))

        # Get ranks of all parens
        candidate_ranks = [
            paren_ranks.get(f"{n}-paren-rank", float("inf")) for n in range(1, 5)
        ]

        min_rank = min(candidate_ranks)

        for i, threshold in enumerate(thresholds):
            if is_promoted(max_logit, subtask_logit, threshold):
                n_total[i] += 1
                if subtask_rank == min_rank:
                    accuracy[i] += 1

    return [accuracy[i] / n_total[i] if n_total[i] > 0 else 0 for i in range(len(thresholds))]


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
            if subtask_rank == min_rank:
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

def get_mlp_neurons(result_dir, model_name):
    """
    Get the neurons to be analyzed from the results directory.
    """
    result_path = os.path.join(result_dir, f"{model_name}_paren_neurons.json")
    neurons_data = read_json(result_path)
    neurons = []
    for each in neurons_data:
        neurons.append((each["layer"], each["neuron_idx"]))
    return neurons  

def process_mlp_data(proj_results_path, result_path, thresholds, neurons, subtask_paren, paren_token_ids):
    subtask_n_paren = subtask_paren.count(")")
    activated_mlps = {}
    for layer, neuron_idx in neurons:
        neuron_name = f"L{layer}N{neuron_idx}"
        data_path = os.path.join(proj_results_path, f"{neuron_name}_proj.json")
        data = read_json(data_path)
        data, precision_data = get_data(data_path, subtask_paren)

        # Initialize metrics
        precision, recall, f1_scores, false_positive_rates = calculate_precision_recall_f1(thresholds, precision_data, subtask_n_paren)


        # accuracy = calculate_accuracy(thresholds, data, subtask_paren, paren_token_ids)
        accuracy, avg_correct_conf, avg_incorrect_conf = calculate_accuracy_and_confidence(thresholds, data, subtask_paren, paren_token_ids, use_second_highest=True)

        activated_mlps[neuron_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_scores,
            "false_positive_rate": false_positive_rates,
            "accuracy": accuracy,
            "avg_correct_conf": avg_correct_conf,
            "avg_incorrect_conf": avg_incorrect_conf
        }

    result_file = os.path.join(result_path, f"{subtask_n_paren}_results.json")
    save_file(activated_mlps, result_file)
    print(f"Results saved to {result_file}")

    del activated_mlps, precision, recall, f1_scores, false_positive_rates, accuracy
    clear_cache()
    time.sleep(0.1)


def main():
    models = read_json("utils/models.json") 
    models = models[-3:] # only run for the last 3 models
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        print(f"Running experiments for {model_name}")
        model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        proj_results_path = f"results/proj_experiment/projections/{folder_name}/final_v/mlp"
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9] # threholds for calculating accuracy, precision, recall, f1

        neurons = get_mlp_neurons(proj_results_path, folder_name)
        proj_results_path = f"results/proj_experiment/projections/{folder_name}/final_v/mlp/proj"
        result_path = f"results/proj_experiment/mlp_results/{folder_name}/proj"
        create_results_dir(result_path)

        paren_token_ids = get_paren_logit_idx(folder_name)


        for paren in tqdm(paren_token_ids):
            subtask_paren = model.tokenizer.decode(paren)
            process_mlp_data(proj_results_path, result_path, thresholds, neurons, subtask_paren=subtask_paren, paren_token_ids=paren_token_ids)       
        
        # macro F1-score, average precision and recall
        # analyze_macro_f1_results(result_path, paren_token_ids, model, thresholds)
        # print(f"Finished experiments for {model_name}")

        # # accuracy generalization
        # accuracy_generalization(result_path, paren_token_ids, model, threshold_value=0.7)

        if model_name == "EleutherAI/pythia-6.9b":
            pdb.set_trace()

        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.1)

if __name__ == "__main__":
    main()