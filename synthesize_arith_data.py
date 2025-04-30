import random
import json
import pdb

def synthesize_sum_dataset(n_classes=20, range_min=51, range_max=999, samples_per_class=50):
    random.seed(42)
    selected_sums = sorted(random.sample(range(range_min, range_max + 1), n_classes))
    dataset = []

    for target_sum in selected_sums:
        valid_pairs = [(a, target_sum - a) for a in range(1, target_sum) if 1 <= target_sum - a <= 1000]
        sampled_pairs = random.sample(valid_pairs, min(samples_per_class, len(valid_pairs)))
        
        for a, b in sampled_pairs:
            dataset.append({
                "prompt": f"{a} + {b} =",
                "answer": str(target_sum),
                "label": target_sum
            })

    return dataset, selected_sums

# Generate dataset
dataset, selected_sums = synthesize_sum_dataset()

# Save to file
output_path = "data/arithmetic/train.json"
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)


output_path, selected_sums[:5]  # Show the file path and first few selected target sums
