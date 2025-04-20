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
    last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
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

    max_logit = logits.max()
    target_logit = logits[token_index]
    
    return target_logit >= threshold * max_logit


def process_comp(model, comp_activations, results, correct_idx):
    # Project activations through the unembedding matrix
    with torch.no_grad():
        logit_projection = comp_activations @ model.W_U  # Shape: (batch, seq_len, vocab_size)

    results["full_logit_entropy"] = logit_entropy(logit_projection[0, -1, :]).item()



    paren_token_ids = [29897, 876, 4961, 13697]
    logits = [logit_projection[0, -1, i].item() for i in paren_token_ids]
    results["paren_entropy"] = logit_entropy(logits).item()


    max_idx = torch.argmax(torch.tensor(logits)).item()
    results["predicted_paren_id"] = paren_token_ids[max_idx]
    results["predicted_paren"] = ")" * (max_idx + 1)
    

    results['paren_ranks'] = {
        "1-paren-rank": get_logit_rank(logit_projection, 29897),
        "2-paren-rank": get_logit_rank(logit_projection, 876),
        "3-paren-rank": get_logit_rank(logit_projection, 4961),
        "4-paren-rank": get_logit_rank(logit_projection, 13697)
    }

    results["paren_logits"] = {
        "1-paren-logit": logit_projection[0, -1, 29897].item(),
        "2-paren-logit": logit_projection[0, -1, 876].item(),
        "3-paren-logit": logit_projection[0, -1, 4961].item(),
        "4-paren-logit": logit_projection[0, -1, 13697].item()
    }
    

    # logit difference between top-1 and top-2, sort and take difference
    paren_logits = [results["paren_logits"][f"{i}-paren-logit"] for i in range(1, 5)]
    sorted_logits = sorted(paren_logits, reverse=True)
    results["logit_diff"] = round(abs(sorted_logits[0] - sorted_logits[1]), 3)

    results["is_correct"] = 1 if results["predicted_paren"] == results["label"] else 0

    results["head_l2_norm"] = torch.linalg.norm(comp_activations[0, -1, :]).item()

    return results

def process_attention_head(model, cache, clean_input, correct_idx, layer, head):
    """Process attention activations for a specific attention head."""
    results = {
        "prompt": clean_input,
        "label": model.tokenizer.decode(correct_idx),
        "label_idx": int(correct_idx[0]),
        "layer": layer,
        "head": head
    }

    # Extract activations for the specific attention head
    attn_activations = cache[f"blocks.{layer}.attn.hook_result"][:, :, head, :]  # Shape: (batch, seq_len, d_model)

    comp_results = process_comp(model, attn_activations, results, correct_idx)
    return comp_results


def attn_proj_experiment(model, data, heads, results_path):
    # data = read_json(data_path)
    dataset = MyDatasetV2(data)
    dataloader = dataset.to_dataloader(batch_size=1)
    # Initialize accuracy tracking
    # Store results for each attention head
    all_results = {head: [] for head in heads}
    for clean_input, correct_idx in tqdm(dataloader):
        tokens = model.to_tokens(clean_input, prepend_bos=True).to(DEVICE)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for layer, head in heads:  # Process each attention head separately
            # Compute results for the attention head
            results = process_attention_head(model, cache, clean_input, correct_idx, layer, head)
            # Store results in list
            all_results[(layer, head)].append(results)
            # Update accuracy tracking
    
    for layer, head in heads:
        results_dir = os.path.join(results_path, "proj")
        # results_dir = results_path + "proj/"
        create_results_dir(results_dir)
        RESULTS_PATH = os.path.join(results_dir, f"L{layer}_H{head}_proj.json")
        save_file(all_results[(layer, head)], RESULTS_PATH)
    

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
    random.seed(42)
    clear_cache()
    models = read_json("utils/models.json")
    # models = models[-3:-2]
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        print(f"Running experiments for {model['name']}")
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        results_dir = f"results/proj_experiment/projections/{folder_name}"
        create_results_dir(results_dir)
        model = load_model(model_name, cache_dir)

        # load the data
        data = []
        for n in range(n_paren):
            data_path = f"{data_dir}/train_labeled_last_paren_{n}.json"
            data += read_json(data_path)
        
        
        # load the heads
        HEADS = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]

        results_path = f"{results_dir}/V2/attn/"
        create_results_dir(results_path)

        attn_proj_experiment(model, data, HEADS, results_path=results_path)

        
        


if __name__ == "__main__":
    main()
