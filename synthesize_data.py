import random
import pdb
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import csv
import torch
import os

from utils.general_utils import load_model

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

def save_file(data, file_name, save_csv=False):
    # json
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data saved in {file_name}!!")

    #csv
    if save_csv:
        csv_file = file_name.replace(".json", ".csv")
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Input data must be a list of dictionaries.")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
            # Create a CSV writer object
            writer = csv.DictWriter(cf, fieldnames=data[0].keys())
            
            # Write the header row
            writer.writeheader()

            # Write the data rows
            writer.writerows(data)


        print(f"Data saved in {file_name} and {csv_file}!!")
    else:
        print(f"Data saved in {file_name}!!")

def base_prompts(n_paren, results_dir, n_samples=500, save_file=False):
    sample_int = [100, 1000]
    all_prompts = {}
    for n in range(n_paren): 
        print(f"Preparing data for {n} parenthesis")
        random.seed(20)
        random_numbers = random.sample(range(sample_int[0], sample_int[1]), n_samples)
        prompts = []
        for num in random_numbers:
            prompt = f"#print the string {num}\nprint(" + "str("*(n) + f"{num})" + ")"*(n)
            prompts.append(prompt)
        
        if save_file:
            save_file(prompts, f"{results_dir}/prompts_{n}.json")
        all_prompts[f"paren_{n}"] = prompts
    return all_prompts

def get_corrupt_prompts(all_prompts, n_paren, results_dir, save_results=False):
    corrupt_prompts = {}
    for n in range(n_paren):
        paren_corrupt_prompts = []
        prompts = all_prompts[f"paren_{n}"]
        for prompt in prompts:
            tmp_corrupt_prompts = []
            
            for c in range(n+1):
                if c == 0:
                    to_be_replace_str = "print("
                    replace_str = "print((" 
                else:
                    to_be_replace_str = f"print(" + "str("*c
                    replace_str = f"print(" + "str("*c +"("
                corrupt_prompt = prompt.replace(to_be_replace_str, replace_str) + ")"
                tmp_corrupt_prompts.append(corrupt_prompt)
            paren_corrupt_prompts.append(tmp_corrupt_prompts)
        corrupt_prompts[f"paren_{n}"] = paren_corrupt_prompts
        if save_results:
            save_file(paren_corrupt_prompts, f"{results_dir}/corrupt_prompts_{n}.json")
    return corrupt_prompts



def label_prompt(all_prompts, tokenizer, results_dir, n_paren=10):
    
    for n in range(n_paren):
        prompts = all_prompts[f"paren_{n}"]
        label_prompts = {}
        label_prompts["last_paren"] = []
        label_prompts["second_last_paren"] = []
        label_prompts["third_last_paren"] = []
        for prompt in prompts:
            tmp_data = {}
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            n_closing_paren = len(tokenizer.tokenize(")"*(n+1), add_special_tokens=False))

            tmp_data["prompt"] = tokenizer.decode(tokenized_prompt["input_ids"][0][:-1], skip_special_tokens=True)
            tmp_data["label"] = tokenizer.decode(tokenized_prompt["input_ids"][0][-1], skip_special_tokens=True)
            tmp_data["label_idx"] = int(tokenized_prompt["input_ids"][0][-1])

            label_prompts["last_paren"].append(tmp_data)
            
            if n_closing_paren == 3: # because codellama tokenizer adds whitespace in front of token
                tmp_data = {}
                tmp_data["prompt"] = tokenizer.decode(tokenized_prompt["input_ids"][0][:-2], skip_special_tokens=True)
                tmp_data["label"] = tokenizer.decode(tokenized_prompt["input_ids"][0][-2], skip_special_tokens=True)
                tmp_data["label_idx"] = int(tokenized_prompt["input_ids"][0][-2])

                label_prompts["second_last_paren"].append(tmp_data)
            
            elif n_closing_paren == 4:
                tmp_data = {}
                tmp_data["prompt"] = tokenizer.decode(tokenized_prompt["input_ids"][0][:-3], skip_special_tokens=True)
                tmp_data["label"] = tokenizer.decode(tokenized_prompt["input_ids"][0][-3], skip_special_tokens=True)
                tmp_data["label_idx"] = int(tokenized_prompt["input_ids"][0][-3])

                label_prompts["third_last_paren"].append(tmp_data)

                tmp_data = {}
                tmp_data["prompt"] = tokenizer.decode(tokenized_prompt["input_ids"][0][:-2], skip_special_tokens=True)
                tmp_data["label"] = tokenizer.decode(tokenized_prompt["input_ids"][0][-2], skip_special_tokens=True)
                tmp_data["label_idx"] = int(tokenized_prompt["input_ids"][0][-2])
                label_prompts["second_last_paren"].append(tmp_data)

        save_file(label_prompts["last_paren"], f"{results_dir}_labeled_last_paren_{n}.json")

        if len(label_prompts["second_last_paren"]) > 1:
            save_file(label_prompts["second_last_paren"], f"{results_dir}_labeled_second_last_paren_{n}.json")
        if len(label_prompts["third_last_paren"]) > 1:
            save_file(label_prompts["third_last_paren"], f"{results_dir}_labeled_third_last_paren_{n}.json")


def get_early_check_tok_prompts(all_prompts, tokenizer, n_paren, results_dir):
    data = []
    for n in range(n_paren):
        if n == 0:
            continue
        else:
            data.extend(all_prompts[f"paren_{n}"])
    #shuffle the data
    random.shuffle(data)

    data = data[:250]
    results = []
    for each in data:
        tmp_data = {}
        try:
            n_closing_paren = each.count(")") # count the number of closing parenthesis
        except:
            pdb.set_trace()
        
        tokenized_prompt = tokenizer(")"*n_closing_paren, return_tensors="pt")
        true_label = tokenizer.decode(tokenized_prompt["input_ids"][0][2], skip_special_tokens=True)
        true_label_num = true_label.count(")")

        # Get a random number between 1 and n_closing_paren, excluding true_label_num
        while True:
            if n_closing_paren < 4:
                random_num = random.randint(1, n_closing_paren-1)
            else:
                random_num = random.randint(1, 4)
            if random_num != true_label_num:
                break
        tmp_data["prompt"] = each[:-n_closing_paren] + ")" *random_num
        tmp_data["label"] = ")"*(n_closing_paren - random_num)
        results.append(tmp_data)

    save_file(results, f"{results_dir}/check_early_tok_prompts.json")


def get_check_tok_prompts(all_prompts, tokenizer, n_paren, results_dir):
    data = []
    for n in range(n_paren):
        if n == 0:
            continue
        else:
            data.extend(all_prompts[f"paren_{n}"])
    #shuffle the data
    random.shuffle(data)

    data = data[:250]
    results = []
    for each in data:
        tmp_data = {}
        try:
            n_closing_paren = each.count(")") # count the number of closing parenthesis
        except:
            pdb.set_trace()
        
        tokenized_prompt = tokenizer(each, return_tensors="pt")
        true_label = tokenizer.decode(tokenized_prompt["input_ids"][0][-1], skip_special_tokens=True)
        true_label_num = true_label.count(")")

        # Get a random number between 1 and n_closing_paren, excluding true_label_num
        while True:
            if n_closing_paren < 4:
                random_num = random.randint(1, n_closing_paren)
            else:
                random_num = random.randint(1, 4)
            if random_num != true_label_num:
                break
        tmp_data["prompt"] = each[:-random_num]
        tmp_data["label"] = each[-random_num:]
        tmp_data["true_label"] = true_label
        results.append(tmp_data)

    save_file(results, f"{results_dir}/check_tok_prompts.json")

def get_intervene_prompts(all_prompts, corrupt_prompts, tokenizer, n_paren, results_dir, save_results=False):
    all_intervene_prompts = []
    for n in range(n_paren):
        prompts = all_prompts[f"paren_{n}"]
        corrupts = corrupt_prompts[f"paren_{n}"]
        intervene_prompts = []
        # counter to sequentially select the corrupt prompt
        n_opening_paren = prompts[0].count("(")
        n_opening_paren_counter = 0 
        for p, c in zip(prompts, corrupts):
            tmp_data = {}
            tokenized_clean_prompt = tokenizer(p, return_tensors="pt")
            tmp_data["clean"] = tokenizer.decode(tokenized_clean_prompt["input_ids"][0][:-1], skip_special_tokens=True)
            tmp_data["correct_idx"] = int(tokenized_clean_prompt["input_ids"][0][-1])
            tmp_data["clean_label"] = tokenizer.decode(tokenized_clean_prompt["input_ids"][0][-1], skip_special_tokens=True)

            if n_opening_paren_counter == n_opening_paren:
                n_opening_paren_counter = 0
            tokenized_corrupt_prompt = tokenizer(c[n_opening_paren_counter], return_tensors="pt")
            tmp_data["corrupted"] = tokenizer.decode(tokenized_corrupt_prompt["input_ids"][0][:-1], skip_special_tokens=True)
            tmp_data["incorrect_idx"] = int(tokenized_corrupt_prompt["input_ids"][0][-1])
            tmp_data["corrupt_label"] = tokenizer.decode(tokenized_corrupt_prompt["input_ids"][0][-1], skip_special_tokens=True)

            # try:
            #     assert len(tokenized_clean_prompt["input_ids"][0]) == len(tokenized_corrupt_prompt["input_ids"][0])
            # except:
            #     continue
            
            intervene_prompts.append(tmp_data)
            n_opening_paren_counter += 1
        save_file(intervene_prompts, f"{results_dir}/intervene_prompts_{n}.json", save_csv=True)

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def train_test_split(all_prompts, n_paren, data_dir, test_size=0.3, save_file=False):
    train_prompts = {}
    test_prompts = {}
    for n in range(n_paren):
        random.shuffle(all_prompts[f"paren_{n}"])
        # Split the data based on test_size
        split_index = int(len(all_prompts[f"paren_{n}"]) * test_size)
        train_prompts[f"paren_{n}"] = all_prompts[f"paren_{n}"][:-split_index]
        test_prompts[f"paren_{n}"] = all_prompts[f"paren_{n}"][-split_index:]
        if save_file:
            save_file(train_prompts[f"paren_{n}"], f"{data_dir}/train_prompts_{n}.json")
            save_file(test_prompts[f"paren_{n}"], f"{data_dir}/test_prompts_{n}.json")
    
    return train_prompts, test_prompts
        
def main():
    # Variables Change
    models = read_json("utils/models.json")
    n_paren = 10
    for model in models:
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        create_results_dir(data_dir)
        model = load_model(model_name, cache_dir)
        tokenizer = model.tokenizer

        all_prompts = base_prompts(n_paren, data_dir, save_file=False)
        train_prompts, test_prompts = train_test_split(all_prompts, n_paren, data_dir)

        # Labeled Train Prompts
        train_labeled_prompts = label_prompt(train_prompts, tokenizer, data_dir+"/train", n_paren)
        # Labeled Test Prompts
        test_labeled_prompts = label_prompt(test_prompts, tokenizer, data_dir+"/test", n_paren)

        
        # labeled_prompts = label_prompt(all_prompts, tokenizer, 10)
        # corrupt_prompts = get_corrupt_prompts(all_prompts, 10)
        # intervene_prompts = get_intervene_prompts(all_prompts, corrupt_prompts, tokenizer, 10)

        # early_check_tok_prompts = get_early_check_tok_prompts(all_prompts, tokenizer, 10)
        # check_tok_prompts = get_check_tok_prompts(all_prompts, tokenizer, 10)


        
    


if __name__ == "__main__":
    main()
