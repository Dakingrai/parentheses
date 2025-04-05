import os
import pandas as pd
import pdb
import json
import csv

import transformer_lens as lens
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model_architecture(model, results_path="results/model_architecture.txt"):
    """
    Function to get the model architecture

    Arguments:
    model (torch.nn.Module): model
    results_path (str): path to save the model architecture
    """
    # get named modules of the model
    with open(results_path, 'w') as f:
        for name, module in model.named_modules():
            f.write("*************"*5 + "\n\n\n")
            f.write(f"{name}\n")
            f.write("-------------------\n")
            f.write(f"{module}\n\n")

    print(f"Model architecture saved in {results_path}!!")
    

def save_file(data, file_name, save_csv=False):
    # json
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

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


def collate_data(xs):
    clean, correct_idx = zip(*xs)
    clean = list(clean)
    # correct_idx and incorrect_idx are tuples containing the clean and
    # corrupted token ids for each respective data example in the batch
    return clean, correct_idx

class MyDataset(Dataset):
    def __init__(self, filepath, num_samples):
        if filepath.endswith(".csv"):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith(".json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data file format ({filepath})")
 

        if num_samples > 0:
            self.df = self.df.sample(n=num_samples, random_state=20)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # correct_idx and incorrect_idx are the token_ids
        return row['prompt'], row['label_idx']
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_data)

# def load_data(data_path: str):
#     """
#     Function to load the data

#     Arguments:
#     data_path (str): path to the data
#     """
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Data file ({data_path}) not found")
    
#     return MyDataset(data_path)


def load_data(data_path: str):
    """
    Function to load the data

    Arguments:
    data_path (str): path to the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file ({data_path}) not found")
    
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data file format ({data_path})")

    return data

def load_model(model_name: str, cache_dir: str = "/projects/ziyuyao/models/llama3-8b-cache/"):
    """
    Function to load the model

    Arguments:
    model_name (str): name of the model
    cache_dir (str): directory to store/load cache of model. If None then uses the default cache directory.
    """
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # if 'gpt2' in model_name:
    #     model = lens.HookedTransformer.from_pretrained(
    #         model_name,
    #         center_unembed=True,
    #         center_writing_weights=True,
    #         fold_ln=True,
    #         refactor_factored_attn_matrices=True,
    #         cache_dir=cache_dir
    #     )
    #     model.cfg.use_split_qkv_input = True
    #     model.cfg.use_hook_mlp_in = True
    #     # NOTE - if too much memory required, we may not need this
    #     model.cfg.use_attn_result = True
    
    # elif 'llama3' in model_name:
    #     model = lens.HookedTransformer.from_pretrained(
    #         "meta-llama/Meta-Llama-3-8B", 
    #         fold_ln=True, 
    #         center_unembed=True, 
    #         center_writing_weights=True,
    #         cache_dir=cache_dir,
    #         dtype="bfloat16",
    #         device="cuda"
    #     )
    #     model.set_use_split_qkv_input(True)
    #     model.set_use_hook_mlp_in(True)
    if "codellama" in model_name:
        inner_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = lens.HookedTransformer.from_pretrained(
            model_name=model_name, 
            hf_model=inner_model, 
            tokenizer=tokenizer,
            fold_ln=True, 
            center_unembed=True, 
            center_writing_weights=False, 
            device="cuda",
            dtype="float16",
        )
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
        # NOTE - if too much memory required, we may not need this
        model.cfg.use_attn_result = True

    elif "pythia" in model_name or "Llama-2-7b" in model_name or "Llama-3-8b" in model_name:
        model = lens.HookedTransformer.from_pretrained(
            model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            cache_dir=cache_dir
        )
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.use_attn_result = True

    else:
        model = lens.HookedTransformer.from_pretrained(
            model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=True,
            cache_dir=cache_dir
        )
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
        # NOTE - if too much memory required, we may not need this
        model.cfg.use_attn_result = True

    model.eval()
    return model