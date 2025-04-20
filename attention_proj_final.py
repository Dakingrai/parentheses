import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
import os
import gc
import time
import random
import torch.nn.functional as F
import numpy as np

random.seed(42)

# Load necessary utilities
from utils.general_utils import MyDataset, load_model, MyDatasetV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_file(data, file_name):
    """Save data to a JSON file."""
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def get_paren_logit_idx(model_name):
    data_dir = f"data/{model_name}"
    paren_logit_idx = []
    for n in range(4):
        data_path = f"{data_dir}/train_labeled_last_paren_{n}.json"
        data = read_json(data_path)
        paren_logit_idx.append(data[0]["label_idx"]) # one example is enough since all examples have the same label
    return paren_logit_idx

def get_logit_rank(logits, index):
    """Get the rank of a logit value at a specific index."""
    last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
    sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
    rank = (sorted_indices == index).nonzero(as_tuple=True)[0].item() + 1
    return rank

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy of the softmax distribution over logits.
    
    Args:
        logits: torch.Tensor of shape [vocab_size] or [batch_size, vocab_size]
    
    Returns:
        entropy: torch.Tensor of shape [] or [batch_size]
    """
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy
    if len(logits.shape) == 1:
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    else:
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    return entropy

def process_comp(model, comp_activations, results, correct_idx, paren_token_ids):
    with torch.no_grad():
        logit_projection = comp_activations @ model.W_U  # Shape: (batch, seq_len, vocab_size)
    
    pred = logit_projection[0, -1, :].argmax().item()
    results["pred"] = pred
    paren_logits = [logit_projection[0, -1, i].item() for i in paren_token_ids]

    # Compute entropy
    results["full_logit_entropy"] = softmax_entropy(logit_projection[0, -1, :]).item()
    results["paren_entropy"] = softmax_entropy(torch.tensor(paren_logits)).item()


    results['paren_ranks'] = {
        "1-paren-rank": get_logit_rank(logit_projection, paren_token_ids[0]),
        "2-paren-rank": get_logit_rank(logit_projection, paren_token_ids[1]),
        "3-paren-rank": get_logit_rank(logit_projection, paren_token_ids[2]),
        "4-paren-rank": get_logit_rank(logit_projection, paren_token_ids[3]),
    }

    results["paren_logits"] = {
        "max-logit": logit_projection[0, -1, :].max().item(),
        "1-paren-logit": logit_projection[0, -1, paren_token_ids[0]].item(),
        "2-paren-logit": logit_projection[0, -1, paren_token_ids[1]].item(),
        "3-paren-logit": logit_projection[0, -1, paren_token_ids[2]].item(),
        "4-paren-logit": logit_projection[0, -1, paren_token_ids[3]].item()
    }

    results["head_l2_norm"] = torch.linalg.norm(comp_activations[0, -1, :]).item()

    return results

def process_attention_head(model, cache, clean_input, correct_idx, layer, head, paren_token_ids):
    """Process attention activations for a specific attention head."""
    results = {
        "prompt": clean_input,
        "label": model.tokenizer.decode(correct_idx),
        "label_idx": correct_idx,
        "layer": layer,
        "head": head,
        "paren_token_ids": paren_token_ids
    }

    # Extract activations for the specific attention head
    attn_activations = cache[f"blocks.{layer}.attn.hook_result"][:, :, head, :]  # Shape: (batch, seq_len, d_model)

    comp_results = process_comp(model, attn_activations, results, correct_idx, paren_token_ids)
    return comp_results

def attn_proj_experiment(model, data, heads, paren_token_ids, results_path):
    dataset = MyDatasetV2(data)
    dataloader = dataset.to_dataloader(batch_size=1)
    # Store results for each attention head
    all_results = {head: [] for head in heads}
    for clean_input, correct_idx in tqdm(dataloader):
        correct_idx = int(correct_idx[0])
        tokens = model.to_tokens(clean_input, prepend_bos=True).to(DEVICE)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for layer, head in heads:  # Process each attention head separately
            # Compute results for the attention head
            results = process_attention_head(model, cache, clean_input, correct_idx, layer, head, paren_token_ids)
            # Store results in list
            all_results[(layer, head)].append(results)
            # Update accuracy tracking
    
    for layer, head in heads:
        results_dir = os.path.join(results_path, "proj")
        # results_dir = results_path + "proj/"
        create_results_dir(results_dir)
        RESULTS_PATH = os.path.join(results_dir, f"L{layer}_H{head}_proj.json")
        save_file(all_results[(layer, head)], RESULTS_PATH)

def main():
    clear_cache()
    models = read_json("utils/models.json")
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        print(f"Running experiments for {model['name']}")
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        results_dir = f"results/proj_experiment/projections/{folder_name}/final_v/attn"
        create_results_dir(results_dir)
        model = load_model(model_name, cache_dir)

        # load the data
        data = []
        for n in range(n_paren):
            data_path = f"{data_dir}/train_labeled_last_paren_{n}.json"
            data += read_json(data_path)
        paren_token_ids = get_paren_logit_idx(folder_name)
        # load the heads
        HEADS = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]

        attn_proj_experiment(model, data, HEADS, paren_token_ids, results_path=results_dir)

        
if __name__ == "__main__":
    main()