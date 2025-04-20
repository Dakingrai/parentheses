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

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load necessary utilities
from utils.general_utils import MyDataset, load_model
from activation_patching import InterveneOV, InterveneNeurons


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_file(data, file_name):
    """Save data to a JSON file."""
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)
    

def analyze_ff_neurons(result_dir, n_paren=4):

    neurons = defaultdict(list)
    for n in range(n_paren):
        result_dir_n = os.path.join(result_dir, f"{n}_neuron_acc.json")
        data = read_json(result_dir_n)
        for each in data:
            neurons[each["neuron"]].append(each)

    # Step 2: Compute metrics for each neuron
    neuron_metrics = {}
    for neuron_id, records in neurons.items():
        TP = sum([each["true_positive"] for each in records])
        FP = sum([each["false_positive"] for each in records])
        FN = sum([each["false_negative"] for each in records])
        TN = sum([each["true_negative"] for each in records])

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        neuron_metrics[neuron_id] = {
            "true_positive": TP,
            "false_positive": FP,
            "false_negative": FN,
            "true_negative": TN,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }


def analyze_attn_head_projections(model, result_dir):
    heads = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]

    head_results = defaultdict(list)
    for layer, head in tqdm(heads):
        result_dir_n = os.path.join(result_dir, f"L{layer}_H{head}_proj.json")
        data = read_json(result_dir_n)
        
        # subtask level analysis
        head_results[f"L{layer}_H{head}"] = {}
        for n in range(1, 5):
            true_label = ")"*n

            # Accuracy = total correct on this subtask / total examples for this subtask
            n_correct = [each["is_correct"] for each in data if each["label"] == true_label]
            accuracy = sum(n_correct) / len(n_correct) if len(n_correct) > 0 else 0

            # Confidence on correct predictions
            n_confidence = [each["logit_diff"] for each in data if each["label"] == true_label and each["predicted_paren"] == true_label]
            confidence_score = sum(n_confidence) / len(n_confidence) if len(n_confidence) > 0 else 0

            n_confidence_strength = [
                each["head_l2_norm"] for each in data if each["label"] == true_label and each["predicted_paren"] == true_label
            ]
            confidence_strength = sum(n_confidence_strength) / len(n_confidence_strength) if len(n_confidence_strength) > 0 else 0

            # Interference = confidence on incorrect predictions that predict this class
            n_interference_score = [
                each["logit_diff"]
                for each in data
                if each["label"] != true_label and each["predicted_paren"] == true_label
            ]
            interference_score = sum(n_interference_score) / len(n_interference_score) if len(n_interference_score) > 0 else 0 

            # L2 norm on those wrong predictions (optional)
            n_interference_strength = [
                each["head_l2_norm"] for each in data
                if each["label"] != true_label and each["predicted_paren"] == true_label
            ]
            interference_strength = sum(n_interference_strength) / len(n_interference_strength) if len(n_interference_strength) > 0 else 0          

            # precision, recall, f1
            true_positive = [1 for each in data if each["label"] == true_label and each["predicted_paren"] == true_label]
            false_negative = [1 for each in data if each["label"] == true_label and each["predicted_paren"] != true_label]

            false_positive = [1 for each in data if each["label"] != true_label and each["predicted_paren"]== true_label]
            
            prec = sum(true_positive) / (sum(true_positive) + sum(false_positive)) if (sum(true_positive) + sum(false_positive)) > 0 else 0
            rec = sum(true_positive) / (sum(true_positive) + sum(false_negative)) if (sum(true_positive) + sum(false_negative)) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            all_metrics = {
                "accuracy": accuracy,
                "confidence_score": confidence_score,
                "interference_score": interference_score,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confidence_strength_l2": confidence_strength,
                "interference_strength_l2": interference_strength,
            }

            head_results[f"L{layer}_H{head}"][f"subtask_{n}"] = all_metrics
        
        # Global metrics
        macro_f1 = sum([
            head_results[f"L{layer}_H{head}"][f"subtask_{n}"]["f1_score"]
            for n in range(1, 5)
        ]) / 4

        # confidence on correct predictions
        n_confidence = [each["logit_diff"] for each in data if each["is_correct"] == 1]
        confidence_score = sum(n_confidence) / len(n_confidence) if len(n_confidence) > 0 else 0

        n_confidence_strength = [
            each["head_l2_norm"] for each in data if each["is_correct"] == 1
        ]
        confidence_strength = sum(n_confidence_strength) / len(n_confidence_strength) if len(n_confidence_strength) > 0 else 0

        # Confidence on incorrect predictions
        n_interference_score = [
            each["logit_diff"] for each in data if each["is_correct"] == 0
        ]
        interference_score = sum(n_interference_score) / len(n_interference_score) if len(n_interference_score) > 0 else 0
        n_interference_strength = [
            each["head_l2_norm"] for each in data if each["is_correct"] == 0
        ]
        interference_strength = sum(n_interference_strength) / len(n_interference_strength) if len(n_interference_strength) > 0 else 0

        # classify if the head is 1-subtask, 2-subtask, 3-subtask, or 4-subtask generlizable if the accuracy is higher than 0.7
        acc_threshold = 0.7
        subtask_acc = [head_results[f"L{layer}_H{head}"][f"subtask_{n}"]["accuracy"] for n in range(1, 5)]
        n_generalized_subtasks = sum(acc > acc_threshold for acc in subtask_acc)

        
        # Store global metrics
        head_results[f"L{layer}_H{head}"]["global"] = {
            "macro_f1": macro_f1,
            "confidence_score": confidence_score,
            "interference_score": interference_score,
            "confidence_strength_l2": confidence_strength,
            "interference_strength_l2": interference_strength,
            "n_generalized_subtasks": n_generalized_subtasks,
        }


    # Save the results
    results_dir = os.path.join(result_dir, "head_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, "head_results.json")
    save_file(head_results, results_path)

    del head_results
        
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
        
def accuracy_analysis(result_dir, model_name, n_paren=4):
    # Step 1: Load the data
    n_heads = [5, 10, 20, 30,  40,  50, 60]
    coeffs = [1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
    results = {}
    for n in range(n_paren):
        results[f"{n}-paren"] = {}
        for n_head in tqdm(n_heads): 
            results[f"{n}-paren"][n_head] = 0
            for c in tqdm(coeffs):
                result_path = f"{result_dir}/{n}_{c}_attn_results_1_general_{n_head}.json"
                acc_results = read_json(result_path)
                if acc_results["accuracy"] > results[f"{n}-paren"][n_head]:
                    results[f"{n}-paren"][n_head] = round(acc_results["accuracy"], 3)
    
    save_file(results, f"temp_results/{model_name}.json")
                

def plot_results(result_path, model_name):
    # Re-defining the data after kernel reset
    results = read_json(result_path)

    # Plotting
    plt.figure(figsize=(10, 6))

    for label, data in results.items():
        x = [int(k) for k in data.keys()]
        y = list(data.values())
        plt.plot(x, y, marker='o', label=label)

    plt.xlabel("Number of Promoted Attention Heads")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Promoted Attention Heads per Sub-task")
    plt.legend(title="Sub-task")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"temp_results/{model_name}.png")

def count_neurons(model_name):
    print(f"Counting neurons for {model_name}")
    results_dir = f"results/proj_experiment/projections/{model_name}/last_paren/neurons"
    results = {}
    for n in range(1, 5):
        results[f"{n}-paren"] = []
        results[f"{n}-paren-all"] = []
        result_dir_n = os.path.join(results_dir, f"{n}_neuron_acc.json")
        data = read_json(result_dir_n)
        for each in data:
            results[f"{n}-paren-all"].append(each)
            if each["f1_score"] > 0.1:
                results[f"{n}-paren"].append(each)
    
    for n in range(1, 5):
        print(f"{n}-paren: {len(results[f'{n}-paren'])}")
        print(f"{n}-paren-all: {len(results[f'{n}-paren-all'])}")
    pdb.set_trace()

def main():
    models = read_json("utils/models.json")
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)

        folder_name = m["name"].split("/")[-1]

        count_neurons(model_name)


        # result_dir = f"results/proj_experiment/improve_performance/V2/{folder_name}/last_paren/new"
        # accuracy_analysis(result_dir, model_name, n_paren=4)
        # plot_results(f"temp_results/{model_name}.json", model_name)




        # result_attn_dir = f"results/proj_experiment/projections/{folder_name}/V2/attn/proj"
        # analyze_attn_head_projections(model, result_attn_dir)

        # clear_cache()
        # del model


        # result_neuron_dir = f"results/proj_experiment/projections/{folder_name}/last_paren/neurons"
        
        # analyze_ff_neurons(result_neuron_dir, n_paren=4)


if __name__ == "__main__":
    main()

