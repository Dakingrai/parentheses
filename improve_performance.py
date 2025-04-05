import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc
from collections import defaultdict

# Load necessary utilities
from utils.general_utils import MyDataset, load_model
from activation_patching import InterveneOV, InterveneNeurons


# # Constants
# MODEL_NAME = "codellama/CodeLlama-7b-hf"
# CACHE_DIR = "../../../projects/ziyuyao/models/codellama/"

# MODEL_NAME = "meta-llama/Llama-2-7b"
# CACHE_DIR = "../../../projects/ziyuyao/models/Llama-2-7b/"


# Constants
# MODEL_NAME = "gpt2-small"
# CACHE_DIR = "../../../models/gpt2-small"

# MODEL_NAME = "gpt2-medium"
# CACHE_DIR = "../../../models/gpt2-medium"

# MODEL_NAME = "EleutherAI/pythia-6.9b"
# CACHE_DIR = "../../../projects/ziyuyao/models/pythia-6.9b"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_file(data, file_name):
    """Save data to a JSON file."""
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)


def get_heads_rank(HEADS, attn_results_path, n_paren=0):
    rank_results = []
    for layer, head in HEADS:
        data = read_json(f"{attn_results_path}/proj/{n_paren}_L{layer}_H{head}_proj.json")
        total_logit_diff = 0
        count = 0
        for each in data:
            total_logit_diff += abs(each["min-logit-diff"])
            count += 1
        avg_logit_diff = round(total_logit_diff / count, 3)
        rank_results.append((layer, head, avg_logit_diff))
    rank_results.sort(key=lambda x: x[2], reverse=True)

    return rank_results

def get_heads(model, head_file_path, n_paren=4):
    head_names = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    correct_heads = {"1-paren": [], "2-paren": [], "3-paren": [], "4-paren": []}
    for i in range(n_paren):
        proj_results = read_json(f"{head_file_path}/{i}_acc.json")
        for layer, head in head_names:
            try:
                truth_value = proj_results[f"L{layer}H{head}"]
                if truth_value > 0.8:
                    correct_heads[(f"{i+1}-paren")].append((layer, head))
            except:
                pass
    
    # sort the heads by min-logit-diff
    correct_heads["1-paren-sorted"] = get_heads_rank(correct_heads["1-paren"], head_file_path, n_paren=0)
    correct_heads["2-paren-sorted"] = get_heads_rank(correct_heads["2-paren"], head_file_path, n_paren=1)
    correct_heads["3-paren-sorted"] = get_heads_rank(correct_heads["3-paren"], head_file_path, n_paren=2)
    correct_heads["4-paren-sorted"] = get_heads_rank(correct_heads["4-paren"], head_file_path, n_paren=3)
    
    all_heads = correct_heads["1-paren-sorted"] + correct_heads["2-paren-sorted"] + correct_heads["3-paren-sorted"] + correct_heads["4-paren-sorted"]
    rank_all_heads = defaultdict(float)
    for layer, head, value in all_heads:
        rank_all_heads[f"{layer}-{head}"] += value


    correct_heads["1-2-paren"] = list(set(correct_heads["1-paren"]) & set(correct_heads["2-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["1-2-paren"]]
    correct_heads["1-2-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["1-3-paren"] = list(set(correct_heads["1-paren"]) & set(correct_heads["3-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["1-3-paren"]]
    correct_heads["1-3-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["1-4-paren"] = list(set(correct_heads["1-paren"]) & set(correct_heads["4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["1-4-paren"]]
    correct_heads["1-4-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["2-3-paren"] = list(set(correct_heads["2-paren"]) & set(correct_heads["3-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["2-3-paren"]]
    correct_heads["2-3-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["2-4-paren"] = list(set(correct_heads["2-paren"]) & set(correct_heads["4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["2-4-paren"]]
    correct_heads["2-4-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["3-4-paren"] = list(set(correct_heads["3-paren"]) & set(correct_heads["4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["3-4-paren"]]
    correct_heads["3-4-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["1-2-3-paren"] = list(set(correct_heads["1-paren"]) & set(correct_heads["2-paren"]) & set(correct_heads["3-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["1-2-3-paren"]]
    correct_heads["1-2-3-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["1-2-4-paren"] = list(set(correct_heads["1-paren"]) & set(correct_heads["2-paren"]) & set(correct_heads["4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["1-2-4-paren"]]
    correct_heads["1-2-4-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["1-3-4-paren"] = list(set(correct_heads["1-paren"]) & set(correct_heads["3-paren"]) & set(correct_heads["4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["1-3-4-paren"]]
    correct_heads["1-3-4-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["2-3-4-paren"] = list(set(correct_heads["2-paren"]) & set(correct_heads["3-paren"]) & set(correct_heads["4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["2-3-4-paren"]]
    correct_heads["2-3-4-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["1-2-3-4-paren"] = list(set(correct_heads["1-paren"]) & set(correct_heads["2-paren"]) & set(correct_heads["3-paren"]) & set(correct_heads["4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["1-2-3-4-paren"]]
    correct_heads["1-2-3-4-paren-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["4-general"] = list(set(correct_heads["1-2-3-4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["4-general"]]
    correct_heads["4-general-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["3-general"] = list(set(correct_heads["1-2-3-paren"] + correct_heads["1-3-4-paren"] + correct_heads["2-3-4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["3-general"]]
    correct_heads["3-general-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["2-general"] = list(set(correct_heads["1-2-paren"] + correct_heads["1-3-paren"] + correct_heads["1-4-paren"] + correct_heads["2-3-paren"] + correct_heads["2-4-paren"] + correct_heads["3-4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["2-general"]]
    correct_heads["2-general-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    correct_heads["1-general"] = list(set(correct_heads["1-paren"] + correct_heads["2-paren"] + correct_heads["3-paren"] + correct_heads["4-paren"]))
    correct_heads_with_rank = [(layer, head, rank_all_heads[f"{layer}-{head}"]) for layer, head in correct_heads["1-general"]]
    correct_heads["1-general-sorted"] = sorted(correct_heads_with_rank, key=lambda x: x[2], reverse=True)

    return correct_heads

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)



def attention_intervention(model, attn_results_path, data_dir, results_dir, n_paren=4):
    HEADS = get_heads(model, attn_results_path)
    # HEADS = list(set(HEADS["4-general"] + HEADS["3-general"] + HEADS["2-general"])) 
    # HEADS = list(set(HEADS["1-general"]))
    n_heads = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    for n_head in tqdm(n_heads): 
        HEADS = list(set(HEADS["1-general-sorted"][:n_head] + HEADS["3-general-sorted"] + HEADS["2-general-sorted"]))
        HEADS = [(layer, head) for layer, head, value in HEADS]
        print(f"Number of heads: {len(HEADS)}")

        coeffs = [1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
        for c in tqdm(coeffs):
            for n in range(n_paren):
                results = {}
                last_data_path = f"{data_dir}/test_labeled_last_paren_{n}.json"
                last_results_path = f"{results_dir}/last_paren/new"
                create_results_dir(last_results_path)
                data = read_json(last_data_path)
                total = 0
                correct = 0
                detail_results = []
                for each in data:
                    total += 1
                    tmp = {}
                    with InterveneOV(model, intervene_heads=HEADS, coeff = c):
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
                save_file(results, f"{last_results_path}/{n}_{c}_attn_results_1_general_{n_head}.json")
                # remove the cache and garbage collection
                del logits
                del results
                del detail_results
                del data
                del tmp
                gc.collect()
                torch.cuda.empty_cache()


def get_intervene_neurons(mlp_results_path, n_paren=4):
    paren_neurons = []
    for n in range(n_paren):
        neurons = read_json(f"{mlp_results_path}/{n}_neuron_acc.json")
        for n in neurons:
            if n["accuracy"] > 0.8:
                paren_neurons.append(n["neuron"])
    return paren_neurons


def mlp_intervention(model, mlp_results_path, data_dir, results_dir, n_paren=4):
    intervene_neurons = get_intervene_neurons(mlp_results_path, n_paren=n_paren)
    print(f"Number of neurons: {len(intervene_neurons)}")
    coeffs = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    for c in tqdm(coeffs):
        for n in range(n_paren):
            results = {}
            last_data_path = f"{data_dir}/test_labeled_last_paren_{n}.json"
            last_results_path = f"{results_dir}/last_paren/neurons"
            create_results_dir(last_results_path)
            data = read_json(last_data_path)
            total = 0
            correct = 0
            detail_results = []
            for each in data:
                total += 1
                tmp = {}
                with InterveneNeurons(model, intervene_neurons=intervene_neurons, coeff = c):
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
            save_file(results, f"{last_results_path}/{n}_{c}_mlp_results.json")
            
            # remove the cache and garbage collection
            del logits
            del results
            del detail_results
            del data
            del tmp
            gc.collect()
            torch.cuda.empty_cache()

def main():
    models = read_json("utils/models.json")
    models = models
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        attn_results_path = f"results/proj_experiment/projections/{folder_name}/last_paren"
        mlp_results_path = f"results/proj_experiment/projections/{folder_name}/last_paren/neurons"

        results_dir = f"results/proj_experiment/improve_performance/{folder_name}"
        create_results_dir(results_dir)

        model = load_model(model_name, cache_dir)

        attention_intervention(model, attn_results_path, data_dir, results_dir, n_paren=n_paren)
        # mlp_intervention(model, mlp_results_path, data_dir, results_dir, n_paren=4)


if __name__ == "__main__":
    main()






