import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc
from collections import defaultdict
import time
import matplotlib.pyplot as plt

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

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)

def parse_attn_name(name):
    """
    Parse the attention name to get the layer and head number.
    """
    name = name.split("L")
    layer = int(name[1].split("_")[0])
    head = name[1].split("_")[1]
    head = int(head.split("H")[1])
    return (layer, head)

def get_heads(attn_results_path, metric = "f1-score", index = 0):

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
    
    head_attn_path = f"{attn_results_path}/macro_metrics.json"
    head_attn = read_json(head_attn_path)

    one_attns = []
    for each in one_general_heads:
        attn = head_attn[each]
        if metric == "f1-score":
            one_attns.append((each, attn["macro_f1"][0]))
        elif metric == "precision":
            one_attns.append((each, attn["average_precision"][0]))
        elif metric == "recall":
            one_attns.append((each, attn["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")

    one_attns = sorted(one_attns, key=lambda x: x[1], reverse=True)
    one_attns = [attn[0] for attn in one_attns]

    two_attns = []
    for each in two_general_heads:
        attn = head_attn[each]
        if metric == "f1-score":
            two_attns.append((each, attn["macro_f1"][0]))
        elif metric == "precision":
            two_attns.append((each, attn["average_precision"][0]))
        elif metric == "recall":
            two_attns.append((each, attn["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    two_attns = sorted(two_attns, key=lambda x: x[1], reverse=True)
    two_attns = [attn[0] for attn in two_attns]

    three_attns = []
    for each in three_general_heads:
        attn = head_attn[each]
        if metric == "f1-score":
            three_attns.append((each, attn["macro_f1"][0]))
        elif metric == "precision":
            three_attns.append((each, attn["average_precision"][0]))
        elif metric == "recall":
            three_attns.append((each, attn["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    three_attns = sorted(three_attns, key=lambda x: x[1], reverse=True)
    three_attns = [attn[0] for attn in three_attns]

    four_attns = []
    for each in four_general_heads:
        attn = head_attn[each]
        if metric == "f1-score":
            four_attns.append((each, attn["macro_f1"][0]))
        elif metric == "precision":
            four_attns.append((each, attn["average_precision"][0]))
        elif metric == "recall":
            four_attns.append((each, attn["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    four_attns = sorted(four_attns, key=lambda x: x[1], reverse=True)
    four_attns = [attn[0] for attn in four_attns]

    all_attns = four_attns + three_attns + two_attns + one_attns

    all_attns = [parse_attn_name(attn) for attn in all_attns]
    
    return all_attns


def load_general_heads(path: str) -> dict:
    general_data = read_json(path)
    general_groups = general_data.get("per_subtask_heads", {})

    result = {}
    result["all_heads"] = []
    for level in ["1-paren", "2-paren", "3-paren", "4-paren"]:
        result[level] = general_groups[level]
        result["all_heads"].extend(general_groups[level])
    
    result["all_heads"] = list(set(result["all_heads"]))
    return result


def extract_scores(heads: list[str], head_attn: dict, metric: str) -> list[str]:
    metric_map = {
        "f1-score": "macro_f1",
        "precision": "average_precision",
        "recall": "average_recall"
    }

    if metric not in metric_map:
        raise ValueError(f"Invalid metric: {metric}")
    
    metric_key = metric_map[metric]
    scored_heads = [
        (head, head_attn[head][metric_key][0]) for head in heads if head in head_attn
    ]
    
    scored_heads.sort(key=lambda x: x[1], reverse=True)
    return [head for head, _ in scored_heads]

def calculate_metrics(attns, attn_metrics):
    precisions = []
    recalls = []
    f1_scores = []
    for attn_name in attns:
        precisions.append(attn_metrics[attn_name]["average_precision"][0])
        recalls.append(attn_metrics[attn_name]["average_recall"][0])
        f1_scores.append(attn_metrics[attn_name]["macro_f1"][0])
    

    results = {
        "precisions": sorted(precisions, reverse=True),
        "recalls": sorted(recalls, reverse=True),
        "f1_scores": sorted(f1_scores, reverse=True)
    }
    return results

def get_heads_v2(attn_results_path: str, results_path: str) -> list[tuple[int, int, int]]:
    macro_metrics_path = f"{attn_results_path}/macro_metrics.json"
    head_attn = read_json(macro_metrics_path)

    # save the scores by subtask
    head_generalization = load_general_heads(f"{attn_results_path}/head_generalization_0.json")
    all_head_metric = calculate_metrics(head_generalization["all_heads"], head_attn)
    one_paren_metric = calculate_metrics(head_generalization["1-paren"], head_attn)
    two_paren_metric = calculate_metrics(head_generalization["2-paren"], head_attn)
    three_paren_metric = calculate_metrics(head_generalization["3-paren"], head_attn)
    four_paren_metric = calculate_metrics(head_generalization["4-paren"], head_attn)

    

    # Save the scores to a file
    results = {
        "all_heads": all_head_metric,
        "1_paren": one_paren_metric,
        "2_paren": two_paren_metric,
        "3_paren": three_paren_metric,
        "4_paren": four_paren_metric
    }

    
    save_file(results, f"{results_path}/metric_scores.json")

    return results

def plot_figures(data, results_path: str):
    # Define metrics and filenames
    metrics = {
        'precision': data['precisions'][:60],
        'recall': data['recalls'][:60],
        'f1_score': data['f1_scores'][:60]
    }

    for name, values in metrics.items():
        plt.figure()
        plt.hist(values, bins=30)
        plt.title(f'{name.replace("_", " ").capitalize()} Distribution')
        plt.xlabel(name.replace("_", " ").capitalize())
        plt.ylabel('Frequency')
        filename = f'{results_path}-{name}-60.png'
        plt.savefig(filename)
        

def main():
    models = read_json("utils/models.json")
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        model_name = model["name"]
        print(f"Model name: {model_name}")
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        attn_results_path = f"results/proj_experiment/attn_results/{folder_name}/final_v"

        results_dir = f"results/proj_experiment/attn_results/{folder_name}/final_v"
        create_results_dir(results_dir)

        
        heads_stat = get_heads_v2(attn_results_path, results_dir)
        
        result_name = f"{results_dir}/all"
        plot_figures(heads_stat["all_heads"], result_name)

        result_name = f"{results_dir}/1_paren"
        plot_figures(heads_stat["1_paren"], result_name)

        result_name = f"{results_dir}/2_paren"
        plot_figures(heads_stat["2_paren"], result_name)

        result_name = f"{results_dir}/3_paren"
        plot_figures(heads_stat["3_paren"], result_name)

        result_name = f"{results_dir}/4_paren"
        plot_figures(heads_stat["4_paren"], result_name)

        del model
        clear_cache()


if __name__ == "__main__":
    main()