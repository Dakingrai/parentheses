# Re-import libraries after reset
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pdb

# Load the F1-based accuracy data again
filename = "f1_both_accuracy"
file_path = Path(f"results/temp_results/{filename}.json")
with open(file_path, "r") as f:
    f1_data = json.load(f)

# Define sub-tasks and model list
sub_tasks = ["0-paren", "1-paren", "2-paren", "3-paren"]
models = list(f1_data.keys())

# Generate and save the plots
output_paths = []
for sub_task in sub_tasks:
    plt.figure(figsize=(10, 6))
    for model in models:
        x = []
        y = []
        for head, score in f1_data[model][sub_task].items():
            head_int = int(head)
            if head_int <= 60:
                x.append(head_int)
                y.append(score)
        if x:
            x, y = zip(*sorted(zip(x, y)))
            plt.plot(x, y, marker='o', label=model)

    plt.title(f"F1 Accuracy vs Number of Heads (<=60) for {sub_task}")
    plt.xlabel("Number of Heads")
    plt.ylabel("F1 Accuracy")
    plt.ylim(0, 1.00)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = f"results/temp_results/{filename}_{sub_task}.png"
    plt.savefig(output_file)
    plt.close()
    output_paths.append(output_file)

