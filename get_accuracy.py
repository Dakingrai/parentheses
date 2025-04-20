import pdb
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from utils.general_utils import load_model

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_file(data, file_name, save_csv=False):
    # json
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def get_accuracy(data, model, result_path):
    total = 0
    correct = 0
    prompt_results = []
    for each in tqdm(data):
        total += 1
        with torch.no_grad():
            logits = model(each["prompt"], return_type="logits")
        l = logits.argmax(dim=-1).squeeze()[-1]
        pred = model.to_string(l)
        tmp = {}
        tmp['prompt'] = each["prompt"]
        tmp['label'] = each["label"]
        tmp['pred'] = pred

        if pred == each["label"]:
            tmp['correct'] = True
            correct += 1
        else:
            tmp['correct'] = False


        prompt_results.append(tmp)
    
    results = {}
    results["accuracy"] = correct/total
    results["prompt_results"] = prompt_results
    save_file(results, result_path)
    
def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def main():

    models = read_json("utils/models.json")
    models = models[-3:-2]
    n_depth = 4
    for model in models:
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"indent-data/{folder_name}"
        results_dir = f"indent-results/acc/{folder_name}"
        create_results_dir(results_dir)
        model = load_model(model_name, cache_dir)

        for n in range(1, n_depth+1):
            data = read_json(f"{data_dir}/train_prompts_depth_{n}.json")
            get_accuracy(data, model, result_path = f"{results_dir}/train_last_paren_{n}.json")
            data = read_json(f"{data_dir}/test_prompts_depth_{n}.json")
            get_accuracy(data, model, result_path = f"{results_dir}/test_last_paren_{n}.json")
        
        
if __name__ == "__main__":
    main()