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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")

def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def generate_one_completion(prompt, model):
    # Add a prefix to the prompt
    prompt = prompt.replace("    ", "\t") # with this the accuracy was updated from 0.0548 to 0.292
    toks = model.to_tokens(prompt)
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
    return generated_code


def main():
    problems = read_problems()
    num_samples_per_task = 1
    models = read_json("utils/models.json")
    models = models[3:]

    for m in models:
        print(f"Running experiments for {m['name']}")
        model_name = m["name"]
        cache_dir = m["cache"]
        folder_name = m["name"].split("/")[-1]
        results_dir = f"results/human_evals/accuracy/{folder_name}"
        create_results_dir(results_dir)
        model = load_model(model_name, cache_dir)
        
        samples = [
            dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"], model))
            for task_id in tqdm(problems)
            for _ in range(num_samples_per_task)
        ]

        write_jsonl(f"{results_dir}/samples.jsonl", samples)
        
        del model, samples
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)


if __name__ == "__main__":
    main()