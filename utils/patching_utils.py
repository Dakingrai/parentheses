import torch
import pandas as pd

# attention head selection helper functions
def select_topk_att_heads(nodes_effect: torch.Tensor, num_layers: int, total_num_heads: int, k: int = 5):
    flat_heads = nodes_effect.flatten()
    values, indicies = torch.topk(flat_heads, k)
    head_layers = indicies // total_num_heads
    head_indicies = indicies % total_num_heads
    intervene_heads_t = torch.stack((head_layers, head_indicies), dim=1)
    intervene_heads = [tuple(intervene_heads_t[i].tolist()) for i in range(k)]
    
    print("Attention Heads Selected:")
    print(intervene_heads)
    return intervene_heads

def select_threshold_att_heads(nodes_effect: torch.Tensor, num_layers: int, num_heads, threshold: float = 0.02):
    intervene_heads = []
    for layer in range(num_layers):
        for head in range(num_heads):
            if abs(nodes_effect[layer][head]) > threshold:
                intervene_heads.append((layer, head))
    return intervene_heads

def dep_select_topk_att_heads(nodes_effect: torch.Tensor, k: int, position_independent_ah: list, first_corrupt_pos: int):    
    num_positions = nodes_effect.shape[1]
    flat_heads = nodes_effect.flatten()
    values, indicies = torch.topk(flat_heads, k)
    heads = indicies // num_positions
    head_positions = indicies % num_positions
    intervene_heads = [position_independent_ah[i] for i in heads]
    intervene_positions = [int(i+first_corrupt_pos) for i in head_positions]
    return intervene_heads, intervene_positions

# helper function
def get_first_corruption_pos(model, df: pd.DataFrame):
    # assumption: corruption_indicies are stored as string representations of tuples
    # find corruption_indices label that corresponds to the earliest corrupted open parenthesis
    corruption_labels = df["corruption_indices"].unique()
    try:
        earliest_idx = min([idx_str[idx_str.index(",") - 1] for idx_str in corruption_labels], key=int)
    except (ValueError, IndexError) as e:
        print(f"Unable to get first corruption position due to formatting of corruption_indices column: {e}")
        breakpoint()

    earliest_corruption_label = None
    i = 0
    while not earliest_corruption_label and i < len(corruption_labels):
        if earliest_idx in corruption_labels[i]:
            earliest_corruption_label = corruption_labels[i]
        i+=1

    # find first token position that is corrupted
    example_record = df[df["corruption_indices"] == earliest_corruption_label].iloc[0]
    clean_tokens = model.to_str_tokens(example_record["clean"])
    corrupted_tokens = model.to_str_tokens(example_record["corrupted"])

    earliest_token_pos = -1
    i = 0
    while earliest_token_pos == -1 and i < len(clean_tokens):
        if clean_tokens[i] != corrupted_tokens[i]:
            earliest_token_pos = i
        i+=1

    return earliest_token_pos