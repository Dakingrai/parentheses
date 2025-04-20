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

def get_logit_rank(logits, index):
    """Get the rank of a logit value at a specific index."""
    if len(logits.shape) > 1:
        last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
    else:
        last_token_logits = logits
    sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
    rank = (sorted_indices == index).nonzero(as_tuple=True)[0].item() + 1
    return rank

def logit_entropy(logits: torch.Tensor) -> torch.Tensor:
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

def get_paren_logit_idx(model_name):
    data_dir = f"data/{model_name}/"
    paren_logit_idx = []
    for n in range(4):
        data_path = f"{data_dir}/train_labeled_last_paren_{n}.json"
        data = read_json(data_path)
        paren_logit_idx.append(data[0]["label_idx"])
    return paren_logit_idx

def filter_neurons(W2: torch.Tensor, W_U: torch.Tensor, paren_token_ids: list[int], layer: int, threshold: float = 0.01):
    """
    Filters and labels neurons that promote paren tokens based on static logit threshold (no input).

    Args:
        W2 (torch.Tensor): Second FF layer weight matrix, shape [d_mlp, d_model]
        W_U (torch.Tensor): Unembedding matrix, shape [d_model, vocab_size]
        paren_token_ids (list[int]): Token IDs for closing parens: ")", "))", etc.
        layer (int): The layer index (for metadata)
        threshold (float): Minimum ratio of max logit (default = 0.01)

    Returns:
        List[Dict]: Metadata for each neuron that promotes at least one paren token.
    """
    neuron_logits = W2 @ W_U  # shape: [d_mlp, vocab_size]
    selected_neurons = []

    for neuron_idx in range(W2.shape[0]):
        logit_vec = neuron_logits[neuron_idx]
        max_logit = logit_vec.max().item()

        # Compute logit ratios for each paren token
        paren_logits = {tok: logit_vec[tok].item() for tok in paren_token_ids}
        paren_ranks = {tok: (logit_vec >= logit_vec[tok]).sum().item() for tok in paren_token_ids}

        # Check if any paren token meets the relative logit threshold
        is_promoted = any(logit_vec[tok].item() >= threshold * max_logit for tok in paren_token_ids)

        if is_promoted:
            selected_neurons.append({
                "neuron_idx": neuron_idx,
                "layer": layer,
                "max_logit": max_logit,
                "paren_logits": paren_logits,
                "paren_ranks": paren_ranks,
            })

    return selected_neurons

def save_paren_neurons(model, results_path, paren_token_ids):
    """
    Efficiently identifies MLP neurons that project specific parentheses within a given rank threshold,
    removing explicit loops for faster computation.
    """
    paren_neurons = [] # Get all the neurons that project some parentheses within a threshold rank

    num_layers, num_neurons = model.W_out.shape[:2]
    paren_neurons = []
    for layer in tqdm(range(num_layers)):
        paren_neurons.extend(filter_neurons(model.W_out[layer], model.W_U, paren_token_ids, layer, threshold=0.5))
    
    save_file(paren_neurons, f"{results_path}/{model.cfg.model_name}_paren_neurons.json")
    clear_cache()
    torch.cuda.empty_cache()
    time.sleep(0.1)  # Allow some time for the cache to clear
    return paren_neurons

def process_mlp_neuron(model, cache, clean_input, correct_idx, layer, neuron_idx, paren_token_ids):
    """Process  activations for a specific mlp neuron."""
    results = {
        "prompt": clean_input,
        "label": model.tokenizer.decode(correct_idx),
        "label_idx": correct_idx,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "paren_token_ids": paren_token_ids
    }

    # Extract activations for the specific mlp neuron
    
    neuron_activation_score = cache[f"blocks.{layer}.mlp.hook_post"][:, -1, neuron_idx].item() 
    results["neuron_activation_score"] = neuron_activation_score
    neuron_parameter = model.W_out[layer][neuron_idx]
    with torch.no_grad():
        neuron_contribution = neuron_activation_score * neuron_parameter
        logit_projection = neuron_contribution @ model.W_U  # Shape: (vocab_size,)
    paren_logits = [logit_projection[i].item() for i in paren_token_ids]

    # Compute entropy
    results["full_logit_entropy"] = logit_entropy(logit_projection).item()
    results["paren_entropy"] = logit_entropy(torch.tensor(paren_logits)).item()

    results['paren_ranks'] = {
        "1-paren-rank": get_logit_rank(logit_projection, paren_token_ids[0]),
        "2-paren-rank": get_logit_rank(logit_projection, paren_token_ids[1]),
        "3-paren-rank": get_logit_rank(logit_projection, paren_token_ids[2]),
        "4-paren-rank": get_logit_rank(logit_projection, paren_token_ids[3])
    }


    results["paren_logits"] = {
        "max-logit": logit_projection.max().item(),
        "1-paren-logit": logit_projection[paren_token_ids[0]].item(),
        "2-paren-logit": logit_projection[paren_token_ids[1]].item(),
        "3-paren-logit": logit_projection[paren_token_ids[2]].item(),
        "4-paren-logit": logit_projection[paren_token_ids[3]].item()
    }

    results["head_l2_norm"] = torch.linalg.norm(neuron_contribution).item()

    return results


def neuron_proj_experiment(model, data, neurons, paren_token_ids, results_path):
    # data = read_json(data_path)
    dataset = MyDatasetV2(data)
    dataloader = dataset.to_dataloader(batch_size=1)
    # Initialize accuracy tracking
    # Store results for each attention head
    all_results = {f"L{neuron['layer']}N{neuron['neuron_idx']}": [] for neuron in neurons}
    for clean_input, correct_idx in tqdm(dataloader):
        correct_idx = int(correct_idx[0])
        tokens = model.to_tokens(clean_input, prepend_bos=True).to(DEVICE)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for neuron in neurons:  # Process each attention head separately
            # Compute results for the attention head
            neuron_idx = neuron["neuron_idx"]
            layer = neuron["layer"]
            results = process_mlp_neuron(model, cache, clean_input, correct_idx, layer, neuron_idx, paren_token_ids)
            
            # Store results in list
            all_results[f"L{layer}N{neuron_idx}"].append(results)
            # Update accuracy tracking
    
    for neuron in neurons:
        neuron_idx = neuron["neuron_idx"]
        layer = neuron["layer"]
        results_dir = os.path.join(results_path, "proj")
        # results_dir = results_path + "proj/"
        create_results_dir(results_dir)
        RESULTS_PATH = os.path.join(results_dir, f"L{layer}N{neuron_idx}_proj.json")
        save_file(all_results[f"L{layer}N{neuron_idx}"], RESULTS_PATH)
    
    clear_cache()
    del model, dataset, dataloader, all_results, clean_input, correct_idx, tokens, cache
    torch.cuda.empty_cache()
    time.sleep(0.1)  # Allow some time for the cache to clear

def main():
    clear_cache()
    models = read_json("utils/models.json")
    models = models[-1:]  # Only use the last model for now
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        print(f"Running experiments for {model['name']}")
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        results_dir = f"results/proj_experiment/projections/{folder_name}/final_v/mlp"
        create_results_dir(results_dir)
        model = load_model(model_name, cache_dir)

        # load the data
        data = []
        for n in range(n_paren):
            data_path = f"{data_dir}/train_labeled_last_paren_{n}.json"
            data += read_json(data_path)
        
        paren_token_ids = get_paren_logit_idx(folder_name)
        
        # load the neurons
        print("Loading neurons...")
        paren_neurons = save_paren_neurons(model, results_dir, paren_token_ids)
        print(f"Found {len(paren_neurons)} neurons that project parentheses")
        neuron_proj_experiment(model, data, paren_neurons, paren_token_ids, results_path=results_dir)

        print(f"Finished experiments for {model_name}")
        clear_cache()
        del model
        torch.cuda.empty_cache()
        time.sleep(0.1)
        

        
if __name__ == "__main__":
    main()