import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
import os
import gc
import time
import random

# Load necessary utilities
from utils.general_utils import MyDataset, load_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_file(data, file_name):
    """Save data to a JSON file."""
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)



def main():
    random.seed(42)
    models = read_json("utils/models.json")
    n_paren = 4 # Number of parentheses to consider
    models = models[-1:]
    for model in models:
        print(f"Running experiments for {model['name']}")
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        model = load_model(model_name, cache_dir)
        pdb.set_trace()

if __name__ == "__main__":
    main()