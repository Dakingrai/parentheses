import pdb
import json
import os
from human_eval.data import write_jsonl, read_problems
import torch
from utils.general_utils import load_model

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

def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")

def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def accuracy():
    models = read_json("utils/models.json")
    models = models[3:]
    for model in models:
        print(f"Running experiments for {model['name']}")
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"results/human_evals/accuracy/{folder_name}"
        data_path = f"{data_dir}/samples.jsonl"

        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
        
        samples = [
            dict(task_id=d["task_id"], completion=filter_code(fix_indents(d["completion"])))
            for d in data]

        write_jsonl(f"{data_dir}/post_samples.jsonl", samples)


def main():
    models = read_json("utils/models.json")
    models = models[4:]
    n_heads = [5, 10, 20, 30, 40, 50, 60]
    for model in models:
        print(f"Running experiments for {model['name']}")
        folder_name = model["name"].split("/")[-1]
        # data_dir = f"results/human_evals/intervene/{folder_name}"
        data_dir = f"results/human_evals/{folder_name}"
        for n in n_heads:
            if n == 0:
                coeffs = [1]
            else:
                coeffs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
            for c in coeffs:
                data_path = f"{data_dir}/samples-head-{n}-coeff-{c}.jsonl"
                with open(data_path, "r") as f:
                    data = [json.loads(line) for line in f]
                samples = [
                    dict(task_id=d["task_id"], completion=filter_code(fix_indents(d["completion"])))
                    for d in data]
                write_jsonl(f"{data_dir}/post_samples-head-{n}-coeff-{c}.jsonl", samples)
    

if __name__ == "__main__":
    main()