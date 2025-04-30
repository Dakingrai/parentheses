from human_eval.data import write_jsonl, read_problems
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pdb
import json
from transformer_lens import HookedTransformer
import os
import gc
from collections import defaultdict
import time
import json

# Load necessary utilities
from utils.general_utils import MyDataset, load_model
from activation_patching import InterveneOV, InterveneNeurons, InterveneOV_HF


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def parse_attn_name(name):
    """
    Parse the attention name to get the layer and head number.
    """
    name = name.split("L")
    layer = int(name[1].split("_")[0])
    head = name[1].split("_")[1]
    head = int(head.split("H")[1])
    return (layer, head)


def attention_intervention(model, attn_results_path, data_dir, results_dir, n_paren=4, metric="f1-score"):
    results_dir = f"{results_dir}/{metric}"
    create_results_dir(results_dir)
    ALL_HEADS = get_heads(attn_results_path, metric=metric)
    print(f"Number of heads: {len(ALL_HEADS)}")
    n_heads = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, int(0.8*len(ALL_HEADS)), len(ALL_HEADS)]

    for n_head in tqdm(n_heads):
        HEADS = ALL_HEADS[:n_head]
        if n_head == 0:
            coeffs = [1]
        else:
            coeffs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
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
                save_file(results, f"{last_results_path}/subtask-{n}-coeff-{c}-heads-{n_head}.json")

                print(f"results saved to {last_results_path}/subtask-{n}_coeff-{c}_heads-{n_head}.json")
                # remove the cache and garbage collection
    del model, data, results
    clear_cache()

def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")

def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def generate_one_completion_prev(prompt, model, tokenizer, HEADS, coeff):
    # Add a prefix to the prompt
    prompt = prompt.replace("    ", "\t") # with this the accuracy was updated from 0.0548 to 0.292
    with torch.no_grad():
        with InterveneOV_HF(model, intervene_heads=HEADS, coeff = coeff):
            outputs = model.generate(
                prompt,
                max_new_tokens=1024,  # Increased for complex functions
                do_sample=False,       # Enable sampling
                temperature=0.0,      # this increased accuracy from 0.292 to 0.329
            )
    
    outputs = outputs.split(prompt)[1]
    
    filtered_code = filter_code(fix_indents(outputs)) # with this the accuracy was updated from 0.01 to 0.0548
    return filtered_code


def generate_one_completion(prompt, model, HEADS, c):
    # Add a prefix to the prompt
    prompt = prompt.replace("    ", "\t") # with this the accuracy was updated from 0.0548 to 0.292
    toks = model.to_tokens(prompt)
    with InterveneOV(model, intervene_heads=HEADS, coeff = c):
        with torch.no_grad():
            outputs = model.generate(
                toks,
                max_new_tokens=512,  # Increased for complex functions
                do_sample=False,       # Enable sampling
                temperature=0.0,      # this increased accuracy from 0.292 to 0.329
            )
    # Decode the output tokens
    input_ids_cutoff = toks.size(dim=1)
    generated_output = outputs[:, input_ids_cutoff:]
    generated_code = model.to_string(generated_output[0])
    # filtered_code = filter_code(fix_indents(generated_code)) # with this the accuracy was updated from 0.01 to 0.0548
    del outputs
    return generated_code


def main():
    problems = read_problems()
    num_samples_per_task = 1
    models = read_json("utils/new_models.json")
    models = models[-2:] # running for Llama2
    for model in models:
        print(f"Running experiments for {model['name']}")
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        results_dir = f"results/human_evals/intervene/{folder_name}"
        attn_results_path = f"results/proj_experiment/attn_results/{folder_name}/final_v"
        create_results_dir(results_dir)
        model = load_model(model_name, cache_dir)

        ALL_HEADS = get_heads(attn_results_path, metric="f1-score")
        print(f"Number of heads: {len(ALL_HEADS)}")
        n_heads = [0, 5, 10, 20, 30, 40, 50, 60]
        # n_heads = [10, 20, 30, 40, 50, 60]

        for n_head in tqdm(n_heads):
            HEADS = ALL_HEADS[:n_head]
            if n_head == 0:
                coeffs = [1]
            else:
                coeffs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
            for c in tqdm(coeffs):
                samples = [
                    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"], model, HEADS, c))
                    for task_id in problems
                    for _ in range(num_samples_per_task)
                ]

                write_jsonl(f"{results_dir}/samples-head-{n_head}-coeff-{c}.jsonl", samples)

                del samples
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1.0)

if __name__ == "__main__":
    main()