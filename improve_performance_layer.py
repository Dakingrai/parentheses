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

# Load necessary utilities
from utils.general_utils import MyDataset, load_model
from activation_patching import InterveneOV, InterveneNeurons, InterveneMLPRemove, InterveneAttentionRemove


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

def intervene_mlp(model, data_dir, results_dir, n_paren=4):
    for n in range(n_paren):
        results = {}
        last_data_path = f"{data_dir}/test_labeled_last_paren_{n}.json"
        last_results_path = f"{results_dir}"
        create_results_dir(last_results_path)
        data = read_json(last_data_path)
        total = 0
        correct = 0
        detail_results = []
        for each in data:
            total += 1
            tmp = {}
            with InterveneMLPRemove(model):
                logits = model(each["prompt"], return_type="logits")
            l = logits.argmax(dim=-1).squeeze()[-1]
            pred = model.to_string(l)
            # save data
            tmp['prompt'] = each["prompt"]
            tmp['label'] = each["label"]
            tmp['pred'] = pred
            if pred == each["label"]:
                tmp['correct'] = True
                correct += 1
            else:
                tmp['correct'] = False
            detail_results.append(tmp)
            gc.collect()
            torch.cuda.empty_cache()
        results[f"accuracy"] = correct / total
        results["detail_results"] = detail_results
        save_file(results, f"{last_results_path}/subtask-{n}-ff-remove.json")
        print(f"results saved to {last_results_path}/subtask-{n}-ff-remove.json")
        # remove the cache and garbage collection
    del model, data, results
    clear_cache()

def intervene_attention(model, data_dir, results_dir, n_paren=4):
    for n in range(n_paren):
        results = {}
        last_data_path = f"{data_dir}/test_labeled_last_paren_{n}.json"
        last_results_path = f"{results_dir}"
        create_results_dir(last_results_path)
        data = read_json(last_data_path)
        total = 0
        correct = 0
        detail_results = []
        for each in data:
            total += 1
            tmp = {}
            with InterveneAttentionRemove(model):
                logits = model(each["prompt"], return_type="logits")
            l = logits.argmax(dim=-1).squeeze()[-1]
            pred = model.to_string(l)
            # save data
            tmp['prompt'] = each["prompt"]
            tmp['label'] = each["label"]
            tmp['pred'] = pred
            if pred == each["label"]:
                tmp['correct'] = True
                correct += 1
            else:
                tmp['correct'] = False
            detail_results.append(tmp)
            gc.collect()
            torch.cuda.empty_cache()
        results[f"accuracy"] = correct / total
        results["detail_results"] = detail_results
        save_file(results, f"{last_results_path}/subtask-{n}-attn-remove.json")
        print(f"results saved to {last_results_path}/subtask-{n}-attn-remove.json")
        # remove the cache and garbage collection
    del model, data, results
    clear_cache()


def main():
    models = read_json("utils/models.json")
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        model_name = model["name"]
        print(f"Model name: {model_name}")
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"

        results_dir = f"results/proj_experiment/improve_performance/final_v/{folder_name}/layer"
        create_results_dir(results_dir)

        model = load_model(model_name, cache_dir)

        # intervene_mlp(model, data_dir, results_dir, n_paren=n_paren)
        intervene_attention(model, data_dir, results_dir, n_paren=n_paren)

        del model
        clear_cache()


if __name__ == "__main__":
    main()