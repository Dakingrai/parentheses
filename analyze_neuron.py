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

def is_promoted_rank(ranks, threshold_rank=50):
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
    Calculates precision, recall, F1 score, and false positive rate for a given neuron.
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
            if is_promoted_rank(all_ranks, threshold_rank=100):
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
    if model_name == "Llama-2-7b":
        model_name = "Llama-2-7b-hf"
    if model_name == "gpt2-small":
        model_name = "gpt2"
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

def analyze_macro_f1_results(result_path, paren_token_ids, model, thresholds):
    """
    Computes per-threshold average precision, recall, and F1 for each mlp neuron (aggregated across subtasks).
    Saves a JSON file mapping neuron name to its per-threshold scores.
    """

    # Store cumulative values to compute average per threshold
    neuron_metrics_accumulator = {}

    for paren in paren_token_ids:
        subtask_paren = model.tokenizer.decode(paren)
        subtask_n = subtask_paren.count(")")
        result_file = os.path.join(result_path, f"{subtask_n}_results.json")

        if not os.path.exists(result_file):
            print(f"Warning: Result file {result_file} not found.")
            continue

        all_neurons = read_json(result_file)
        for neuron_name, metrics in all_neurons.items():
            if neuron_name not in neuron_metrics_accumulator:
                neuron_metrics_accumulator[neuron_name] = {
                    "precision": [[] for _ in thresholds],
                    "recall": [[] for _ in thresholds],
                    "f1": [[] for _ in thresholds]
                }

            for i in range(len(thresholds)):
                neuron_metrics_accumulator[neuron_name]["precision"][i].append(metrics["precision"][i])
                neuron_metrics_accumulator[neuron_name]["recall"][i].append(metrics["recall"][i])
                neuron_metrics_accumulator[neuron_name]["f1"][i].append(metrics["f1"][i])

    # Compute average across subtasks for each threshold
    neuron_avg_metrics = {}
    for neuron_name, lists in neuron_metrics_accumulator.items():
        avg_prec = [float(np.mean(p)) for p in lists["precision"]]
        avg_recall = [float(np.mean(r)) for r in lists["recall"]]
        avg_f1 = [float(np.mean(f)) for f in lists["f1"]]

        neuron_avg_metrics[neuron_name] = {
            "average_precision": avg_prec,
            "average_recall": avg_recall,
            "macro_f1": avg_f1
        }

    # Save final result
    save_path = os.path.join(result_path, "macro_metrics.json")
    save_file(neuron_avg_metrics, save_path)
    print(f"\nSaved neuron per-threshold metrics to: {save_path}")

    del neuron_avg_metrics, neuron_metrics_accumulator
    clear_cache()
    time.sleep(0.5)

def accuracy_generalization(result_path, paren_token_ids, model, threshold_value=0.8, threshold_index=0):
    """
    Groups neurons that exceed a given accuracy threshold (e.g. 0.8) at a specific threshold index
    (e.g. index 0 for threshold=0.5). Saves both generalization groups and per-subtask neurons sets
    into a single result JSON file.
    """
    from collections import defaultdict

    neurons_that_pass = defaultdict(set)  

    # Step 1: Collect passing neurons for each subtask
    for paren in paren_token_ids:
        subtask_paren = model.tokenizer.decode(paren)
        subtask_n = subtask_paren.count(")")
        result_file = os.path.join(result_path, f"{subtask_n}_results.json")

        if not os.path.exists(result_file):
            print(f"Warning: Result file {result_file} not found.")
            continue

        all_neurons = read_json(result_file)

        for neuron_name, metrics in all_neurons.items():
            accuracy_list = metrics.get("accuracy", [])
            if accuracy_list[threshold_index] >= threshold_value:
                neurons_that_pass[subtask_n].add(neuron_name)

    # Step 2: Format per-subtask neurons
    per_subtask_output = {
        f"{k}-paren": sorted(list(v)) for k, v in neurons_that_pass.items()
    }

    # Step 3: Build generalization groups (how many subtasks each neuron succeeded on)
    neuron_to_subtasks = defaultdict(set)
    for subtask_id, neuron_set in neurons_that_pass.items():
        for neuron in neuron_set:
            neuron_to_subtasks[neuron].add(subtask_id)

    generalization_groups = defaultdict(list)
    for neuron, subtasks in neuron_to_subtasks.items():
        n = len(subtasks)
        key = f"{n}-general-neurons"
        generalization_groups[key].append(neuron)

    generalization_output = {
        group: {
            "neurons": sorted(neurons),
            "n_len": len(neurons)
        }
        for group, neurons in generalization_groups.items()
    }

    # Final result
    final_output = {
        "generalization_groups": generalization_output,
        "per_subtask_neurons": per_subtask_output
    }

    save_path = os.path.join(result_path, f"neuron_generalization_{threshold_index}.json")
    save_file(final_output, save_path)
    print(f"\nSaved full neuron generalization summary to: {save_path}")

    del final_output, neurons_that_pass, per_subtask_output, generalization_groups
    clear_cache()
    time.sleep(0.5)

def analyze_generalization(result_path, proj_results_path, paren_token_ids, model):
    generalization_neuron_path = os.path.join(result_path, "neuron_generalization_0.json")
    generalization_data = read_json(generalization_neuron_path)

    one_general_neurons = generalization_data["generalization_groups"]["1-general-neurons"]["neurons"]
    try:
        two_general_neurons = generalization_data["generalization_groups"]["2-general-neurons"]["neurons"]
    except:
        two_general_neurons = []
    try:
        three_general_neurons = generalization_data["generalization_groups"]["3-general-neurons"]["neurons"]
    except:
        three_general_neurons = []
    
    try:
        four_general_neurons = generalization_data["generalization_groups"]["4-general-neurons"]["neurons"]
    except:
        four_general_neurons = []
    
    print(f"Number of neurons that generalize to 1 subtask: {len(one_general_neurons)}")
    print(f"Number of neurons that generalize to 2 subtasks: {len(two_general_neurons)}")
    print(f"Number of neurons that generalize to 3 subtasks: {len(three_general_neurons)}")
    print(f"Number of neurons that generalize to 4 subtasks: {len(four_general_neurons)}")

    # For each neuron in two_general_neuron, get the average logit for each subtask
    general_neurons = one_general_neurons + two_general_neurons + three_general_neurons + four_general_neurons
    all_neurons = {}
    for neuron in general_neurons:
        all_neurons[neuron] = {}
        layer, neuron_idx = parse_mlp_name(neuron)
        neuron_proj_results_path = os.path.join(proj_results_path, f"{neuron}_proj.json")
        neuron_data = read_json(neuron_proj_results_path)
        paren_tokens = [model.tokenizer.decode(paren) for paren in paren_token_ids]
        n_count = {}
        all_neurons[neuron]["generalization"] = {}
        all_neurons[neuron]["avg_logits"] = {}
        all_neurons[neuron]["ranks"] = {}
        for paren in paren_tokens:
            all_neurons[neuron]["avg_logits"][paren] = 0.0
            n_count[paren] = 0
            all_neurons[neuron]["generalization"][paren] = False
            all_neurons[neuron]["ranks"][paren] = 0.0

        # take sum of logits for each subtask
        for entry in neuron_data:
            entry_label = entry["label"]
            entry_label_count = entry_label.count(")")
            n_count[entry_label] += 1
            all_neurons[neuron]["avg_logits"][entry_label] += entry["paren_logits"][f"{entry_label_count}-paren-logit"]
            all_neurons[neuron]["ranks"][entry_label] += entry["paren_ranks"][f"{entry_label_count}-paren-rank"]


        # take average of logits for each subtask
        for paren in paren_tokens:
            all_neurons[neuron]["avg_logits"][paren] /= n_count[paren]
            all_neurons[neuron]["ranks"][paren] /= n_count[paren]

            
        
        # Find out on which tasks the neuron has accuracy above threshold
        for paren in paren_tokens:
            n_paren = paren.count(")")
            if neuron in generalization_data["per_subtask_neurons"][f"{n_paren}-paren"]:
                all_neurons[neuron]["generalization"][paren] = True

    # Save the results to a JSON file
    save_path = os.path.join(result_path, "neuron_coeffs.json")
    save_file(all_neurons, save_path)
    print(f"\nSaved neuron generalization summary to: {save_path}")


def parse_mlp_name(name):
    """
    Parse the MLP name to get the layer and neuron number.
    """
    layer = int(name.split("N")[0].split("L")[1])
    neuron = int(name.split("N")[1])
    return (layer, neuron)

def process_proj_results():
    models = read_json("utils/models.json") 
    models = models[:-2] # only run for the last 3 models
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        print(f"Running experiments for {model_name}")
        model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        proj_results_path = f"results/proj_experiment/projections/{folder_name}/final_v1/mlp"
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9] # threholds for calculating accuracy, precision, recall, f1

        neurons = get_mlp_neurons(proj_results_path, folder_name)
        proj_results_path = f"results/proj_experiment/projections/{folder_name}/final_v1/mlp/proj"
        result_path = f"results/proj_experiment/mlp_results/{folder_name}/proj"
        create_results_dir(result_path)

        paren_token_ids = get_paren_logit_idx(folder_name)


        for paren in tqdm(paren_token_ids):
            subtask_paren = model.tokenizer.decode(paren)
            process_mlp_data(proj_results_path, result_path, thresholds, neurons, subtask_paren=subtask_paren, paren_token_ids=paren_token_ids)       
        
        # macro F1-score, average precision and recall
        analyze_macro_f1_results(result_path, paren_token_ids, model, thresholds)
        print(f"Finished experiments for {model_name}")

        # accuracy generalization
        accuracy_generalization(result_path, paren_token_ids, model, threshold_value=0.7)

        # Is two generalization implemented with positive and negative coefficients?
        analyze_generalization(result_path, proj_results_path, paren_token_ids, model)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.1)

def count_neurons(attn_results_path):
    general_neurons_path = f"{attn_results_path}/neuron_generalization_0.json"
    general_neurons = read_json(general_neurons_path)
    one_general_neurons = general_neurons["generalization_groups"]["1-general-neurons"]["neurons"]
    try:
        two_general_neurons = general_neurons["generalization_groups"]["2-general-neurons"]["neurons"]
    except:
        two_general_neurons = []
    try:
        three_general_neurons = general_neurons["generalization_groups"]["3-general-neurons"]["neurons"]
    except:
        three_general_neurons = [] 

    try:
        four_general_neurons = general_neurons["generalization_groups"]["4-general-neurons"]["neurons"]
    except:
        four_general_neurons = []
    
    all_neurons = {}
    all_neurons["1-general-neurons"] = len(one_general_neurons)
    all_neurons["2-general-neurons"] = len(two_general_neurons)
    all_neurons["3-general-neurons"] = len(three_general_neurons)
    all_neurons["4-general-neurons"] = len(four_general_neurons)
    return all_neurons

def plot_figure(coeff_values, result_path):
    # Plot histogram with distinction between positive and negative coefficients
    coeff_values = list(coeff_values.values())

    # Separate positive and negative values
    positive_coeffs = [v for v in coeff_values if v > 0]
    negative_coeffs = [v for v in coeff_values if v < 0]

    # Define bins
    num_bins = int(np.ceil(np.log2(len(coeff_values)) + 1))
    bins = np.histogram_bin_edges(coeff_values, bins=num_bins)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.hist(negative_coeffs, bins=bins, color='red', alpha=0.7, label='Negative Coefficients', edgecolor='black')
    plt.hist(positive_coeffs, bins=bins, color='blue', alpha=0.7, label='Positive Coefficients', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Distribution of Neuron Coefficients (Positive vs Negative)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(result_path)

def main():
    # pdb.set_trace()
    process_proj_results()
    # pdb.set_trace()
    models = read_json("utils/models.json")
    models = models[:-1]
    # process_proj_results()
    # count the number of generalization heads for each model
    model_generalization_neurons = {}
    # f1-score
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        neuron_proj_stat = f"results/proj_experiment/mlp_results/{folder_name}/proj"
        neuron_proj_path = f"results/proj_experiment/projections/{folder_name}/final_v1/mlp/proj"

        neurons = read_json(neuron_proj_stat + "/neuron_generalization_0.json")
        one_neurons = neurons["generalization_groups"]["1-general-neurons"]["neurons"]
        one_neurons_label = {}
        for each in one_neurons:
            if each in neurons["per_subtask_neurons"]["1-paren"]:
                one_neurons_label[each] = ")"
            elif each in neurons["per_subtask_neurons"]["2-paren"]:
                one_neurons_label[each] = "))"
            elif each in neurons["per_subtask_neurons"]["3-paren"]:
                one_neurons_label[each] = ")))"
            elif each in neurons["per_subtask_neurons"]["4-paren"]:
                one_neurons_label[each] = "))))"
            else:
                print(f"Neuron {each} not found in any subtask")
                continue
            
        postive_coeff = 0
        negative_coeff = 0
        all_stats = {}
        for each in one_neurons: 
            path = f"{neuron_proj_path}/{each}_proj.json"
            neuron_proj = read_json(path)
            avg_coeff = 0
            for example in neuron_proj:
                if one_neurons_label[each] == example["label"]:
                    avg_coeff += example["neuron_activation_score"]
            avg_coeff /= len(neuron_proj)
            all_stats[each] = avg_coeff
            if avg_coeff > 0:
                postive_coeff += 1
            else:
                negative_coeff += 1
        print(f"Model: {model_name}, Positive Coeff: {postive_coeff}, Negative Coeff: {negative_coeff}")
        # save the results
        save_path = f"results/proj_experiment/mlp_results/{folder_name}/proj/neuron_coeffs.json"
        save_file(all_stats, save_path)

        plot_figure(all_stats, result_path =f"results/proj_experiment/mlp_results/{folder_name}/proj/neuron_coeffs.png")
            

    #     results = count_neurons(attn_results_path)
    
    #     model_generalization_heads[folder_name] = results
    
    # # Step 2: Save the results
    # results_path = "results/temp_results/generalization_neurons_v2.json"
    # create_results_dir("results/temp_results")
    # save_file(model_generalization_heads, results_path)
    

if __name__ == "__main__":
    main()