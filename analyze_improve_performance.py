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

def get_heads(attn_results_path):
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
    
    all_heads = one_general_heads + two_general_heads + three_general_heads + four_general_heads
    return all_heads

def get_neurons(mlp_results_path, metric = "f1-score", index = 0):

    general_mlps_path = f"{mlp_results_path}/neuron_generalization_0.json"
    general_mlps = read_json(general_mlps_path)
    
    one_general_mlps = general_mlps["generalization_groups"]["1-general-neurons"]["neurons"]
    try:
        two_general_mlps = general_mlps["generalization_groups"]["2-general-neurons"]["neurons"]
    except:
        two_general_mlps = []
    try:
        three_general_mlps = general_mlps["generalization_groups"]["3-general-neurons"]["neurons"]
    except:
        three_general_mlps = [] 

    try:
        four_general_mlps = general_mlps["generalization_groups"]["4-general-neurons"]["neurons"]
    except:
        four_general_mlps = []
    
    mlp_neuron_path = f"{mlp_results_path}/macro_metrics.json"
    mlp_neuron = read_json(mlp_neuron_path)

    one_neurons = []
    for each in one_general_mlps:
        neuron = mlp_neuron[each]
        if metric == "f1-score":
            one_neurons.append((each, neuron["macro_f1"][0]))
        elif metric == "precision":
            one_neurons.append((each, neuron["average_precision"][0]))
        elif metric == "recall":
            one_neurons.append((each, neuron["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")

    one_neurons = sorted(one_neurons, key=lambda x: x[1], reverse=True)
    one_neurons = [neuron[0] for neuron in one_neurons]

    two_neurons = []
    for each in two_general_mlps:
        neuron = mlp_neuron[each]
        if metric == "f1-score":
            two_neurons.append((each, neuron["macro_f1"][0]))
        elif metric == "precision":
            two_neurons.append((each, neuron["average_precision"][0]))
        elif metric == "recall":
            two_neurons.append((each, neuron["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    two_neurons = sorted(two_neurons, key=lambda x: x[1], reverse=True)
    two_neurons = [neuron[0] for neuron in two_neurons]

    three_neurons = []
    for each in three_general_mlps:
        neuron = mlp_neuron[each]
        if metric == "f1-score":
            three_neurons.append((each, neuron["macro_f1"][0]))
        elif metric == "precision":
            three_neurons.append((each, neuron["average_precision"][0]))
        elif metric == "recall":
            three_neurons.append((each, neuron["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    three_neurons = sorted(three_neurons, key=lambda x: x[1], reverse=True)
    three_neurons = [neuron[0] for neuron in three_neurons]

    four_neurons = []
    for each in four_general_mlps:
        neuron = mlp_neuron[each]
        if metric == "f1-score":
            four_neurons.append((each, neuron["macro_f1"][0]))
        elif metric == "precision":
            four_neurons.append((each, neuron["average_precision"][0]))
        elif metric == "recall":
            four_neurons.append((each, neuron["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    four_neurons = sorted(four_neurons, key=lambda x: x[1], reverse=True)
    four_neurons = [neuron[0] for neuron in four_neurons]

    all_neurons = four_neurons + three_neurons + two_neurons + one_neurons

    all_neurons = [parse_mlp_name(neuron) for neuron in all_neurons]

    return all_neurons

def parse_mlp_name(name):
    """
    Parse the MLP name to get the layer and neuron number.
    """
    layer = int(name.split("N")[0].split("L")[1])
    neuron = int(name.split("N")[1])
    return (layer, neuron)


def generate_plots(subtasks, models, data, result_dir, attn_results_path):
    # Generate plots with x-axis limited to 0-60 heads and y-axis limited to 1.00
    for sub_task in subtasks:
        plt.figure(figsize=(10, 6))
        for model in models:
            x = []
            y = []
            for head, acc in data[model][sub_task].items():
                head_int = int(head)
                if head_int <= 60:
                    x.append(head_int)
                    y.append(acc)
            if x:  # Only plot if there are values within the range
                x, y = zip(*sorted(zip(x, y)))
                plt.plot(x, y, marker='o', label=model)

        plt.title(f"Accuracy vs Number of Heads (<=60) for {sub_task}")
        plt.xlabel("Number of Heads")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.00)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def attn_accuracy_analysis(result_dir, attn_results_path, model_name, n_paren=4):
    # Step 1: Load the data
    ALL_HEADS = get_heads(attn_results_path)
    print(f"Number of heads: {len(ALL_HEADS)}")
    n_heads = [0, 5, 10, 20, 30, 40, 50, 60, int(0.8*len(ALL_HEADS)), len(ALL_HEADS)]
    results = {}
    for n in range(n_paren):
        results[f"{n}-paren"] = {}
        for n_head in tqdm(n_heads):
            if n_head == 0:
                coeffs = [1]
            else:
                coeffs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] 
            results[f"{n}-paren"][n_head] = 0
            for c in tqdm(coeffs):
                result_path = f"{result_dir}/subtask-{n}-coeff-{c}-heads-{n_head}.json"
                # result_path = f"{result_dir}/{n}_{c}_attn_results_1_general_{n_head}.json"
                acc_results = read_json(result_path)
                if acc_results["accuracy"] > results[f"{n}-paren"][n_head]:
                    results[f"{n}-paren"][n_head] = round(acc_results["accuracy"], 3)
    return results


def accuracy_analysis_attn(models):
    model_results_f1_score = {}
    model_results_precision = {}
    model_results_recall = {}
    # f1-score
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        result_dir = f"results/proj_experiment/improve_performance/final_v/{folder_name}/attn/f1-score/last_paren/new"
        attn_results_path = f"results/proj_experiment/attn_results/{folder_name}/final_v/"

        results = attn_accuracy_analysis(result_dir, attn_results_path, model_name)
        model_results_f1_score[folder_name] = results
    
    # Step 2: Save the results
    results_path = "results/temp_results/f1_accuracy_attns.json"
    create_results_dir("results/temp_results")
    save_file(model_results_f1_score, results_path)


    # precision ----
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        result_dir = f"results/proj_experiment/improve_performance/final_v/{folder_name}/attn/precision/last_paren/new"
        attn_results_path = f"results/proj_experiment/attn_results/{folder_name}/final_v/"
        

        results = attn_accuracy_analysis(result_dir, attn_results_path, model_name)
        model_results_precision[folder_name] = results
    
    # Step 2: Save the results
    results_path = "results/temp_results/precision_accuracy_attns.json"
    create_results_dir("results/temp_results")
    save_file(model_results_precision, results_path)


    # recall ----
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        result_dir = f"results/proj_experiment/improve_performance/final_v/{folder_name}/attn/recall/last_paren/new"
        attn_results_path = f"results/proj_experiment/attn_results/{folder_name}/final_v/"
        

        results = attn_accuracy_analysis(result_dir, attn_results_path, model_name)
        model_results_recall[folder_name] = results
    
    # Step 2: Save the results
    results_path = "results/temp_results/recall_accuracy_attns.json"
    create_results_dir("results/temp_results")
    save_file(model_results_recall, results_path)


def mlp_accuracy_analysis(result_dir, mlp_results_path, metric, n_paren=4):
    # Step 1: Load the data
    ALL_NEURONS = get_neurons(mlp_results_path, metric)
    print(f"Number of heads: {len(ALL_NEURONS)}")
    n_neurons = [0, 5, 10, 20, 30, 40, 50, 60]
    results = {}
    for n in range(n_paren):
        results[f"{n}-paren"] = {}
        for n_neuron in tqdm(n_neurons):
            if n_neuron == 0:
                coeffs = [1]
            else:
                coeffs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] 
            results[f"{n}-paren"][n_neuron] = 0
            for c in tqdm(coeffs):
                result_path = f"{result_dir}/subtask-{n}-coeff-{c}-neurons-{n_neuron}.json"
                acc_results = read_json(result_path)
                if acc_results["accuracy"] > results[f"{n}-paren"][n_neuron]:
                    results[f"{n}-paren"][n_neuron] = round(acc_results["accuracy"], 3)
    return results


def accuracy_analysis_mlp(models):
    model_results_f1_score = {}
    model_results_precision = {}
    model_results_recall = {}
    # f1-score
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        result_dir = f"results/proj_experiment/improve_performance/final_mlp/{folder_name}/mlp/f1-score/last_paren/new"
        mlp_results_path = f"results/proj_experiment/mlp_results/{folder_name}/proj"

    #     results = mlp_accuracy_analysis(result_dir, mlp_results_path, "f1-score")
    #     model_results_f1_score[folder_name] = results
    
    # # Step 2: Save the results
    # results_path = "results/temp_results/f1_mlp_accuracy.json"
    # create_results_dir("results/temp_results")
    # save_file(model_results_f1_score, results_path)


    # precision ----
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        result_dir = f"results/proj_experiment/improve_performance/final_mlp/{folder_name}/mlp/precision/last_paren/new"

        results = mlp_accuracy_analysis(result_dir, mlp_results_path, "precision")
        model_results_precision[folder_name] = results
    
    # Step 2: Save the results
    results_path = "results/temp_results/precision_mlp_accuracy.json"
    create_results_dir("results/temp_results")
    save_file(model_results_precision, results_path)


    # recall ----
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]
        result_dir = f"results/proj_experiment/improve_performance/final_mlp/{folder_name}/mlp/recall/last_paren/new"
        

        results = mlp_accuracy_analysis(result_dir, mlp_results_path, "recall")
        model_results_recall[folder_name] = results
    
    # Step 2: Save the results
    results_path = "results/temp_results/recall_mlp_accuracy.json"
    create_results_dir("results/temp_results")
    save_file(model_results_recall, results_path)


def both_accuracy_analysis(result_dir, n_paren=4):
    # Step 1: Load the data
    n_neurons = [0, 5, 10, 20, 30, 40, 50, 60]
    results = {}
    for n in range(n_paren):
        results[f"{n}-paren"] = {}
        for n_neuron in tqdm(n_neurons):
            if n_neuron == 0:
                coeffs = [1]
            else:
                coeffs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] 
            results[f"{n}-paren"][n_neuron] = 0
            for c in tqdm(coeffs):
                result_path = f"{result_dir}/subtask-{n}-coeff-{c}-both-{n_neuron}.json"
                acc_results = read_json(result_path)
                if acc_results["accuracy"] > results[f"{n}-paren"][n_neuron]:
                    results[f"{n}-paren"][n_neuron] = round(acc_results["accuracy"], 3)
    return results


def accuracy_analysis_both(models):
    model_results_f1_score = {}
    # f1-score
    for m in models:
        folder_name = m["name"].split("/")[-1]
        result_dir = f"results/proj_experiment/improve_performance/final_v/{folder_name}/both/f1-score/last_paren/new"

        results = both_accuracy_analysis(result_dir)
        model_results_f1_score[folder_name] = results
    
    # Step 2: Save the results
    results_path = "results/temp_results/f1_both_accuracy.json"
    create_results_dir("results/temp_results")
    save_file(model_results_f1_score, results_path)


def main():
    models = read_json("utils/models.json")
    models = models[:-3] 
    # accuracy_analysis_attn(models)
    # accuracy_analysis_mlp(models) 
    accuracy_analysis_both(models)


if __name__ == "__main__":
    main()

