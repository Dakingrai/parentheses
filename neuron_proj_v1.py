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

def get_logit_rank(logits, index):
    """Get the rank of a logit value at a specific index."""
    if len(logits.shape) > 1:
        last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
    else:
        last_token_logits = logits
    sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
    rank = (sorted_indices == index).nonzero(as_tuple=True)[0].item() + 1
    return rank

def get_logit_rank_batch(logit_projections, token_id):
    """
    Computes the rank of a specific token for each neuron's logit projection in a batch.
    """
    clear_cache()
    _, sorted_indices = torch.sort(logit_projections, dim=-1, descending=True)
    
    # Get the rank of token_id for each batch element
    ranks = (sorted_indices == token_id).nonzero(as_tuple=True)
    # Extract ranks for each batch element
    batch_ranks = torch.zeros(logit_projections.shape[0], dtype=torch.long).to(DEVICE)
    batch_ranks[ranks[0]] = ranks[1] + 1  # Adjust for 1-based ranking

    clear_cache()
    del sorted_indices
    del ranks
    
    return batch_ranks

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

def promotes_token(logits, token_index, threshold=0.8):
    """
    Determines whether a token is promoted by a component based on relative logit thresholding.

    Args:
        logits (torch.Tensor or np.ndarray): The full vocab-size logit vector from the component.
        token_index (int): The index of the target token to evaluate (e.g., index of ")").
        threshold (float): Proportion threshold (default = 0.8).

    Returns:
        bool: True if token is promoted, False otherwise.
    """
    # Convert to numpy array if needed
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    else:
        logits = np.array(logits)

    max_logit = logits.max()
    target_logit = logits[token_index]
    
    if target_logit >= threshold * max_logit:
        return True
    else:
        return False


def process_comp(model, comp_activations, results, correct_idx):
    # Project activations through the unembedding matrix
    paren_token_ids = [29897, 876, 4961, 13697]
    with torch.no_grad():
        logit_projection = comp_activations @ model.W_U  # Shape: (batch, seq_len, vocab_size)
    
    paren_logits = [logit_projection[0, -1, i].item() for i in paren_token_ids]

    # Compute entropy
    results["full_logit_entropy"] = logit_entropy(logit_projection[0, -1, :]).item()
    results["paren_entropy"] = logit_entropy(torch.tensor(paren_logits)).item()

    # Does the paren tokens fall within the threshold * max(logit_projection)?
    results["threshold_90"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.9) for i in paren_token_ids]
    results["threshold_80"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.8) for i in paren_token_ids]
    results["threshold_70"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.7) for i in paren_token_ids]
    results["threshold_60"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.6) for i in paren_token_ids]
    results["threshold_50"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.5) for i in paren_token_ids]
    results["threshold_40"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.4) for i in paren_token_ids]
    results["threshold_30"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.3) for i in paren_token_ids]
    results["threshold_20"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.2) for i in paren_token_ids]
    results["threshold_10"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.1) for i in paren_token_ids]
    results["threshold_5"] = [promotes_token(logit_projection[0, -1, :], i, threshold=0.05) for i in paren_token_ids]

    results['paren_ranks'] = {
        "1-paren-rank": get_logit_rank(logit_projection, 29897),
        "2-paren-rank": get_logit_rank(logit_projection, 876),
        "3-paren-rank": get_logit_rank(logit_projection, 4961),
        "4-paren-rank": get_logit_rank(logit_projection, 13697)
    }

    results["paren_logits"] = {
        "max-logit": logit_projection[0, -1, :].max().item(),
        "1-paren-logit": logit_projection[0, -1, 29897].item(),
        "2-paren-logit": logit_projection[0, -1, 876].item(),
        "3-paren-logit": logit_projection[0, -1, 4961].item(),
        "4-paren-logit": logit_projection[0, -1, 13697].item()
    }

    results["head_l2_norm"] = torch.linalg.norm(comp_activations[0, -1, :]).item()

    return results


def get_paren_logit_idx(model_name):
    data_dir = f"data/{model_name}/"
    paren_logit_idx = []
    for n in range(4):
        data_path = f"{data_dir}/train_labeled_last_paren_{n}.json"
        data = read_json(data_path)
        paren_logit_idx.append(data[0]["label_idx"])
    return paren_logit_idx


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
    
    neuron_activation_score = cache[f"blocks.{layer}.mlp.hook_pre"][:, -1, neuron_idx].item() 
    neuron_parameter = model.W_out[layer][neuron_idx]
    with torch.no_grad():
        neuron_contribution = neuron_activation_score * neuron_parameter
        logit_projection = neuron_contribution @ model.W_U  # Shape: (vocab_size,)
    paren_logits = [logit_projection[i].item() for i in paren_token_ids]

    # Compute entropy
    results["full_logit_entropy"] = logit_entropy(logit_projection).item()
    results["paren_entropy"] = logit_entropy(torch.tensor(paren_logits)).item()

    # Does the paren tokens fall within the threshold * max(logit_projection)?
    results["threshold_90"] = [promotes_token(logit_projection, i, threshold=0.9) for i in paren_token_ids]
    results["threshold_80"] = [promotes_token(logit_projection, i, threshold=0.8) for i in paren_token_ids]
    results["threshold_70"] = [promotes_token(logit_projection, i, threshold=0.7) for i in paren_token_ids]
    results["threshold_60"] = [promotes_token(logit_projection, i, threshold=0.6) for i in paren_token_ids]
    results["threshold_50"] = [promotes_token(logit_projection, i, threshold=0.5) for i in paren_token_ids]
    results["threshold_40"] = [promotes_token(logit_projection, i, threshold=0.4) for i in paren_token_ids]
    results["threshold_30"] = [promotes_token(logit_projection, i, threshold=0.3) for i in paren_token_ids]
    results["threshold_20"] = [promotes_token(logit_projection, i, threshold=0.2) for i in paren_token_ids]
    results["threshold_10"] = [promotes_token(logit_projection, i, threshold=0.1) for i in paren_token_ids]
    results["threshold_5"] = [promotes_token(logit_projection, i, threshold=0.05) for i in paren_token_ids]

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


def filter_and_label_neurons_multi_thresh(W2, W_U, paren_token_ids, layer, top_k=50, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """
    Filters and labels neurons that promote paren tokens under multiple min_ratio thresholds.

    Args:
        W2 (torch.Tensor): FF layer W2 matrix, shape [d_mlp, d_model].
        W_U (torch.Tensor): Unembedding matrix, shape [d_model, vocab_size].
        paren_token_ids (list[int]): Token IDs for parens.
        top_k (int): Consider top-k tokens per neuron for filtering.
        thresholds (tuple[float]): Ratio thresholds (e.g., 0.1 for 10%).

    Returns:
        dict: neuron_id → {
            'max_logit': float,
            'paren_logits': {token_id: logit},
            'paren_ranks': {token_id: rank},
            'promoted_logits_10': {token_id: logit},
            'promoted_logits_20': {token_id: logit},
            ...
        }
    """
    vocab_size = W_U.shape[1]
    with torch.no_grad():
        # Compute neuron logits
        neuron_logits = W2 @ W_U
        # neuron_softmax = F.softmax(neuron_logits, dim=-1)

    topk_logits, topk_indices = torch.topk(neuron_logits, k=top_k, dim=-1)

    paren_neurons = []

    for neuron_idx in range(W2.shape[0]):
        neuron_vector = neuron_logits[neuron_idx]  # shape: [vocab_size]
        max_logit = neuron_vector.max().item()
        neuron_topk = topk_indices[neuron_idx]

        all_paren_logits = {}
        all_paren_ranks = {}
        promoted_by_thresh = {f"promoted_logits_{int(t * 100)}": {} for t in thresholds}

        for token_id in paren_token_ids:
            token_logit = neuron_vector[token_id].item()
            token_rank = (neuron_vector >= neuron_vector[token_id]).sum().item()

            # Save for meta-info
            all_paren_logits[token_id] = token_logit
            all_paren_ranks[token_id] = token_rank

            if token_id in neuron_topk:
                for t in thresholds:
                    if token_logit >= t * max_logit:
                        promoted_by_thresh[f"promoted_logits_{int(t * 100)}"][token_id] = token_logit

        # Only store neurons that promoted at least one paren token at any threshold
        if any(promoted_by_thresh[f] for f in promoted_by_thresh):
            
            temp = {
                "neuron_idx": neuron_idx,
                "layer": layer,
                "max_logit": max_logit,
                "paren_logits": all_paren_logits,
                "paren_ranks": all_paren_ranks,
                **promoted_by_thresh
            }
            paren_neurons.append(temp)

    return paren_neurons



def save_paren_neurons(model, results_path, paren_token_ids):
    """
    Efficiently identifies MLP neurons that project specific parentheses within a given rank threshold,
    removing explicit loops for faster computation.
    """
    paren_neurons = [] # Get all the neurons that project some parentheses within a threshold rank

    num_layers, num_neurons = model.W_out.shape[:2]
    paren_neurons = []
    for n in tqdm(range(num_layers)):
        paren_neurons.extend(filter_and_label_neurons_multi_thresh(model.W_out[n], model.W_U, paren_token_ids, layer=n))

        
    save_file(paren_neurons, f"{results_path}/{model.cfg.model_name}_paren_neurons.json")
    clear_cache()
    return paren_neurons


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
    

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def get_data(data_path, n_paren = 4, correct_paren=1):
    data = []
    for n in range(n_paren):
        data_path = f"{data_path}/train_labeled_last_paren_{n}.json"
        if n == correct_paren:
            data += read_json(data_path)
        else:
            tmp_data = read_json(data_path)
            random.shuffle(tmp_data)
            n_samples = int(len(tmp_data)/(n_paren-1))
            data += tmp_data[:n_samples]
    return data


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
        results_dir = f"results/proj_experiment/projections/{folder_name}/V5/mlp"
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
        neuron_proj_experiment(model, data, paren_neurons, paren_token_ids, results_path=results_dir)

        
if __name__ == "__main__":
    main()
