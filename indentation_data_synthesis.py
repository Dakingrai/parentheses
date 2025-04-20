import pdb
import random
import json
import os
import sys
import csv


from transformers import AutoTokenizer

from utils.general_utils import load_model

statements_dict = ["for i in range(10):\n", "try:\n", "for i in range(1, 15):\n", "if k == 2:\n", "while k < 2:\n", "while k>5:\n", "if k<3:\n", "my_function():\n" ]

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

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def get_base_prompts(tokenizer, n_prompts = 300, depth=2):
    # Generate the base prompts
    base_prompts = []
    for i in range(n_prompts):
        random_num = random.randint(0, 1000)
        prompt = f"k={random_num}\n"
        for n in range(depth):
            random_num = random.randint(0, len(statements_dict) - 1)
            prompt += statements_dict[random_num] 
            prompt += "   "*(n+1)
        prompt_id = tokenizer.encode(prompt, return_tensors="pt")
        prompt_label = tokenizer.decode(prompt_id[0][-1])
        prompt = tokenizer.decode(prompt_id[0][:-1], skip_special_tokens=True)
        tmp = {
            "prompt": prompt,
            "label_idx": int(prompt_id[0][-1]),
            "label": prompt_label
        }
        base_prompts.append(tmp)
    return base_prompts
    
def get_intervene_prompts(tokenizer, base_prompts, depth=2):
    # Generate the corrupted prompts
    corrupted_prompts = []
    for each in base_prompts:
        random_num = random.randint(1, depth-1)
        corrupt_prompt = each["clean"].replace("    "*random_num, "    "*random_num+"    ")
        full_prompt_with_label = corrupt_prompt + "    "*(depth+1)
        prompt_id = tokenizer.encode(full_prompt_with_label, return_tensors="pt")
        corrupt_prompt_label = tokenizer.decode(prompt_id[0][-1])
        corrupt_prompt = tokenizer.decode(prompt_id[0][:-1], skip_special_tokens=True)
        each["corrupted"] = corrupt_prompt
        each["corrupt_label"] = corrupt_prompt_label
        each["incorrect_idx"] = int(prompt_id[0][-1])

        corrupted_prompts.append(each)
    return corrupted_prompts

def train_test_split(all_prompts, n_depth, data_dir, test_size=0.3):
    train_prompts = {}
    test_prompts = {}

    random.shuffle(all_prompts)
    # Split the data based on test_size
    split_index = int(len(all_prompts) * test_size)
    train_prompts = all_prompts[:-split_index]
    test_prompts = all_prompts[-split_index:]
    save_file(train_prompts, f"{data_dir}/train_prompts_depth_{n_depth}.json")
    save_file(test_prompts, f"{data_dir}/test_prompts_depth_{n_depth}.json")
    
    return train_prompts, test_prompts


def main():
    models = read_json("utils/models.json")
    models = models[-3:-2]
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"indent-data/{folder_name}"
        create_results_dir(data_dir)
        model = load_model(model_name, cache_dir)
        tokenizer = model.tokenizer

        base_prompts_depth_1 = get_base_prompts(tokenizer, n_prompts = 500, depth=1)

        _ = train_test_split(base_prompts_depth_1, 1, data_dir, test_size=0.3)

        base_prompts_depth_2 = get_base_prompts(tokenizer, n_prompts = 500, depth=2)
        _ = train_test_split(base_prompts_depth_2, 2, data_dir, test_size=0.3)

        base_prompts_depth_3 = get_base_prompts(tokenizer, n_prompts = 500, depth=3)
        _ = train_test_split(base_prompts_depth_3, 3, data_dir, test_size=0.3)

        base_prompts_depth_4 = get_base_prompts(tokenizer, n_prompts = 500, depth=4)
        _ = train_test_split(base_prompts_depth_4, 4, data_dir, test_size=0.3)
    

if __name__ == "__main__":
    main()