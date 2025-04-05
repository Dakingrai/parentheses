import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
import os
import gc
import time

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

def get_logit_rank(logits, index):
    """Get the rank of a logit value at a specific index."""
    last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
    sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
    rank = (sorted_indices == index).nonzero(as_tuple=True)[0].item() + 1
    return rank

def process_comp(model, comp_activations, results, correct_idx):
    # Project activations through the unembedding matrix
    with torch.no_grad():
        logit_projection = comp_activations @ model.W_U  # Shape: (batch, seq_len, vocab_size)

    # Store top-10 predicted tokens
    results["top-10-tokens"] = [model.tokenizer.decode(t) for t in logit_projection.topk(10, dim=-1).indices[0, -1].tolist()]

    # Compute ranking for specific tokens
    results["one-paren-rank"] = get_logit_rank(logit_projection, 29897)
    results["two-paren-rank"] = get_logit_rank(logit_projection, 876)
    results["three-paren-rank"] = get_logit_rank(logit_projection, 4961)
    results["four-paren-rank"] = get_logit_rank(logit_projection, 13697)
    results["one-paren-logit"] = logit_projection[0, -1, 29897].item()
    results["two-paren-logit"] = logit_projection[0, -1, 876].item()
    results["three-paren-logit"] = logit_projection[0, -1, 4961].item()
    results["four-paren-logit"] = logit_projection[0, -1, 13697].item()
    
    # Determine correctness based on ranking
    correct_label = model.tokenizer.decode(correct_idx)
    correct_rank = {
        ")": results["one-paren-rank"] < min(results["two-paren-rank"], results["three-paren-rank"], results["four-paren-rank"]),
        "))": results["two-paren-rank"] < min(results["one-paren-rank"], results["three-paren-rank"], results["four-paren-rank"]),
        ")))": results["three-paren-rank"] < min(results["one-paren-rank"], results["two-paren-rank"], results["four-paren-rank"]),
        "))))": results["four-paren-rank"] < min(results["one-paren-rank"], results["two-paren-rank"], results["three-paren-rank"]),
    }
    results["correct-rank"] = correct_rank.get(correct_label, False)
    ranks = [results["one-paren-rank"], results["two-paren-rank"], results["three-paren-rank"], results["four-paren-rank"]]

    sorted_ranks = sorted(ranks)
    results["min-logit-rank-diff"] = abs(sorted_ranks[0] - sorted_ranks[1])

    sorted_logits = sorted([logit_projection[0, -1, 29897].item(), logit_projection[0, -1, 876].item(), logit_projection[0, -1, 4961].item(), logit_projection[0, -1, 13697].item()], reverse=True)
    results["min-logit-diff"] = round(abs(sorted_logits[0] - sorted_logits[1]), 3)

    return results

def process_attention_head(model, cache, clean_input, correct_idx, layer, head):
    """Process attention activations for a specific attention head."""
    results = {
        "prompt": clean_input,
        "label": model.tokenizer.decode(correct_idx),
        "layer": layer,
        "head": head
    }

    # Extract activations for the specific attention head
    attn_activations = cache[f"blocks.{layer}.attn.hook_result"][:, :, head, :]  # Shape: (batch, seq_len, d_model)

    comp_results = process_comp(model, attn_activations, results, correct_idx)
    return comp_results


def process_attention_layer(model, cache, clean_input, correct_idx, layer):
    """Process attention activations for a specific attention layers."""
    results = {
        "prompt": clean_input,
        "label": model.tokenizer.decode(correct_idx),
        "attn_layer": layer,
    }

    # Extract activations for the specific attention layer
    attn_activations = cache[f"blocks.{layer}.hook_attn_out"]  # Shape: (batch, seq_len, d_model)

    comp_results = process_comp(model, attn_activations, results, correct_idx)
    return comp_results


def attn_proj_experiment(model, data_path, heads, n_paren, results_path):
    # data = read_json(data_path)
    dataset = MyDataset(data_path, 300)
    dataloader = dataset.to_dataloader(batch_size=1)
    # Initialize accuracy tracking
    accuracy_counts = {head: {"correct": 0, "total": 0} for head in heads}
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
            accuracy_counts[(layer, head)]["total"] += 1
            if results["correct-rank"]:
                accuracy_counts[(layer, head)]["correct"] += 1
    

    accuracy_results = {
        f"L{layer}H{head}": accuracy_counts[(layer, head)]["correct"] / accuracy_counts[(layer, head)]["total"]
        for layer, head in heads
    }

    for layer, head in heads:
        if accuracy_results[f"L{layer}H{head}"] > 0.5:
            results_dir = os.path.join(results_path, "proj")
            # results_dir = results_path + "proj/"
            create_results_dir(results_dir)
            RESULTS_PATH = os.path.join(results_dir, f"{n_paren}_L{layer}_H{head}_proj.json")
            # RESULTS_PATH = f"{results_dir}/{n_paren}_{results_path}_L{layer}_H{head}_proj.json"
            save_file(all_results[(layer, head)], RESULTS_PATH)
    
    # accuracy_results["data"] = data_path
    # Save accuracy results

    ACCURACY_PATH = f"{results_path}/{n_paren}_attn_acc.json"
    save_file(accuracy_results, ACCURACY_PATH)


def save_paren_neurons(model, results_path, threshold_rank=50):
    """
    Efficiently identifies MLP neurons that project specific parentheses within a given rank threshold,
    removing explicit loops for faster computation.
    """
    paren_neurons = [] # Get all the neurons that project some parentheses within a threshold rank
    one_paren_token = 29897
    two_paren_token = 876
    three_paren_token = 4961
    four_paren_token = 13697

    paren_tokens = [one_paren_token, two_paren_token, three_paren_token, four_paren_token]
    paren_labels = [")", "))", ")))", "))))"]

    num_layers, num_neurons = model.W_out.shape[:2]
    ranks = torch.zeros(num_layers, len(paren_tokens), num_neurons)
    for n in range(num_layers):
        with torch.no_grad():
            logit_projection = model.W_out[n] @ model.W_U
    
        for k, token in enumerate(paren_tokens):
            n_k_rank = get_logit_rank_batch(logit_projection, token)
            for i in range(num_neurons):
                ranks[n, k, i] = n_k_rank[i]

        min_ranks, _ = torch.min(ranks, dim=-2) 

        clear_cache()
        del logit_projection


    mask = min_ranks <= threshold_rank

    # Get indices of neurons that satisfy the condition
    layer_indices, neuron_indices = torch.where(mask)

    for layer, neuron in zip(layer_indices.tolist(), neuron_indices.tolist()):
        neuron_ranks = ranks[layer, :, neuron]

        # Determine the label based on the minimum rank
        label_index = torch.argmin(neuron_ranks).item()
        label = paren_labels[label_index]

        paren_neurons.append({
            "layer": layer,
            "neuron": neuron,
            "one-paren-rank": ranks[layer, 0, neuron].item(),
            "two-paren-rank": ranks[layer, 1, neuron].item(),
            "three-paren-rank": ranks[layer, 2, neuron].item(),
            "four-paren-rank": ranks[layer, 3, neuron].item(),
            "label": label
        })

    save_file(paren_neurons, f"{results_path}/{model.cfg.model_name}_paren_neurons.json")
    clear_cache()
    del ranks
    return paren_neurons
        

def is_neuron_activated(cache, layer, neuron, threshold_rank=50):
    """Process attention activations for a specific attention head."""
    results = {}

    first_layer_mlp_act = cache[f"blocks.{layer}.mlp.hook_pre"][0][-1]

    # Compute ranking for specific tokens
    neuron_rank = get_logit_rank(first_layer_mlp_act, neuron)

    # clear cache
    clear_cache()
    del cache
    if neuron_rank <= threshold_rank:
        return True
   
    return False

        
def mlp_neuron_proj_experiment(model, paren_neurons, data_path, results_path, n_paren):
    data = read_json(data_path)
    # calculate the accuracy of each neurons in paren_neurons
    results = []
    # Initialize accuracy tracking
    accuracy_counts = {neuron["neuron"]: {"correct": 0, "incorrect": 0} for neuron in paren_neurons}
    # Store results for each neuron
    total = 0
    for each in data:
        total += 1
        with torch.no_grad():
            _, cache = model.run_with_cache(each["prompt"])
        for neuron in paren_neurons:
            is_activated = is_neuron_activated(cache, neuron["layer"], neuron["neuron"], threshold_rank=50)
            if is_activated and each["label"] == neuron["label"]:
                accuracy_counts[neuron["neuron"]]["correct"] += 1
            if is_activated and each["label"] != neuron["label"]:
                accuracy_counts[neuron["neuron"]]["incorrect"] += 1
            
            clear_cache()
            del is_activated
        # clear cache
        clear_cache()
        del cache

    
    for neuron in paren_neurons:
        results.append({
            "neuron": f"L{neuron['layer']}N{neuron['neuron']}",
            "accuracy": accuracy_counts[neuron["neuron"]]["correct"]/total,
            "incorrect_accuracy": accuracy_counts[neuron["neuron"]]["incorrect"]/total,
        })
    save_file(results, f"{results_path}/{n_paren}_neuron_acc.json")
            

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def main():
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
        
        # paren_neurons = save_paren_neurons(model, results_dir, threshold_rank=50)
        # print(f"Number of neurons: {len(paren_neurons)}")
        
        HEADS = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
        start_time = time.time()
        for n in tqdm(range(n_paren)): 
            last_data_path = f"{data_dir}/train_labeled_last_paren_{n}.json"
            last_results_path = f"{results_dir}/last_paren/"
            create_results_dir(last_results_path)

            last_results_path_attn = f"{last_results_path}/attn"
            create_results_dir(last_results_path_attn)
            attn_proj_experiment(model, last_data_path, HEADS, n, results_path=last_results_path)

            # last_results_path_neurons = f"{last_results_path}/neurons"
            # create_results_dir(last_results_path_neurons)
            # mlp_neuron_proj_experiment(model, paren_neurons, last_data_path, results_path=last_results_path_neurons, n_paren=n)
            # clear_cache()
        print(f"Time taken: {time.time() - start_time}")


if __name__ == "__main__":
    main()
