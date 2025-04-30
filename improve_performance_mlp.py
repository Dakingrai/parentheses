import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc
from collections import defaultdict
import time

# Load necessary utilities
from utils.general_utils import MyDataset, load_model
from activation_patching import InterveneOV, InterveneNeurons, InterveneMLPNeuron


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_file(data, file_name):
    """Save data to a JSON file."""
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)


def get_neurons(mlp_results_path, metric = "f1-score", index = 0):

    general_mlps_path = f"{mlp_results_path}/neuron_generalization_0.json"
    general_mlps = read_json(general_mlps_path)
    
    one_general_mlps = general_mlps["generalization_groups"]["1-general-neurons"]["neurons"]
    try:
        two_general_mlps = general_mlps["generalization_groups"]["2-general-neurons"]["neurons"]
    except:
        two_general_mlps = []
    try:
        three_general_mlps = general_mlps["generalization_groups"]["3-general-neurons"]["neurons"]
    except:
        three_general_mlps = [] 

    try:
        four_general_mlps = general_mlps["generalization_groups"]["4-general-neurons"]["neurons"]
    except:
        four_general_mlps = []
    
    mlp_neuron_path = f"{mlp_results_path}/macro_metrics.json"
    mlp_neuron = read_json(mlp_neuron_path)

    one_neurons = []
    for each in one_general_mlps:
        neuron = mlp_neuron[each]
        if metric == "f1-score":
            one_neurons.append((each, neuron["macro_f1"][0]))
        elif metric == "precision":
            one_neurons.append((each, neuron["average_precision"][0]))
        elif metric == "recall":
            one_neurons.append((each, neuron["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")

    one_neurons = sorted(one_neurons, key=lambda x: x[1], reverse=True)
    one_neurons = [neuron[0] for neuron in one_neurons]

    two_neurons = []
    for each in two_general_mlps:
        neuron = mlp_neuron[each]
        if metric == "f1-score":
            two_neurons.append((each, neuron["macro_f1"][0]))
        elif metric == "precision":
            two_neurons.append((each, neuron["average_precision"][0]))
        elif metric == "recall":
            two_neurons.append((each, neuron["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    two_neurons = sorted(two_neurons, key=lambda x: x[1], reverse=True)
    two_neurons = [neuron[0] for neuron in two_neurons]

    three_neurons = []
    for each in three_general_mlps:
        neuron = mlp_neuron[each]
        if metric == "f1-score":
            three_neurons.append((each, neuron["macro_f1"][0]))
        elif metric == "precision":
            three_neurons.append((each, neuron["average_precision"][0]))
        elif metric == "recall":
            three_neurons.append((each, neuron["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    three_neurons = sorted(three_neurons, key=lambda x: x[1], reverse=True)
    three_neurons = [neuron[0] for neuron in three_neurons]

    four_neurons = []
    for each in four_general_mlps:
        neuron = mlp_neuron[each]
        if metric == "f1-score":
            four_neurons.append((each, neuron["macro_f1"][0]))
        elif metric == "precision":
            four_neurons.append((each, neuron["average_precision"][0]))
        elif metric == "recall":
            four_neurons.append((each, neuron["average_recall"][0]))
        else:
            raise ValueError("Invalid metric!")
    
    four_neurons = sorted(four_neurons, key=lambda x: x[1], reverse=True)
    four_neurons = [neuron[0] for neuron in four_neurons]

    all_neurons = four_neurons + three_neurons + two_neurons + one_neurons

    all_neurons = [parse_mlp_name(neuron) for neuron in all_neurons]

    return all_neurons

def parse_mlp_name(name):
    """
    Parse the MLP name to get the layer and neuron number.
    """
    layer = int(name.split("N")[0].split("L")[1])
    neuron = int(name.split("N")[1])
    return (layer, neuron)


def mlp_intervention(model, mlp_results_path, data_dir, results_dir, n_paren=4, metric="f1-score"):
    results_dir = f"{results_dir}/{metric}"
    create_results_dir(results_dir)
    ALL_NEURONS = get_neurons(mlp_results_path, metric=metric)
    print(f"Number of neurons: {len(ALL_NEURONS)}")
    n_neurons = [0, 5, 10, 20, 30, 40, 50, 60, int(0.8*len(ALL_NEURONS)), len(ALL_NEURONS)]

    for n_neuron in tqdm(n_neurons):
        NEURONS = ALL_NEURONS[:n_neuron]
        if n_neuron == 0:
            coeffs = [1]
        else:
            coeffs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        for c in tqdm(coeffs):
            for n in range(n_paren):
                results = {}
                last_data_path = f"{data_dir}/test_labeled_last_paren_{n}.json"
                last_results_path = f"{results_dir}/last_paren/new"
                create_results_dir(last_results_path)
                data = read_json(last_data_path)
                total = 0
                correct = 0
                detail_results = []
                for each in data:
                    total += 1
                    tmp = {}
                    with InterveneMLPNeuron(model, intervene_neurons=NEURONS, coeff = c):
                        logits = model(each["prompt"], return_type="logits")
                    l = logits.argmax(dim=-1).squeeze()[-1]
                    pred = model.to_string(l)
                    # save data
                    tmp['prompt'] = each["prompt"]
                    tmp['label'] = each["label"]
                    tmp['pred'] = pred
                    if pred == each["label"]:
                        tmp['correct'] = True
                        correct += 1
                    else:
                        tmp['correct'] = False
                    detail_results.append(tmp)
                    gc.collect()
                    torch.cuda.empty_cache()
                results[f"accuracy"] = correct / total
                results["detail_results"] = detail_results
                save_file(results, f"{last_results_path}/subtask-{n}-coeff-{c}-neurons-{n_neuron}.json")

                print(f"results saved to {last_results_path}/subtask-{n}_coeff-{c}-neurons-{n_neuron}.json")
                # remove the cache and garbage collection
    del model, data, results
    clear_cache()


def main():
    models = read_json("utils/models.json")
    models = models[-3:]
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        model_name = model["name"]
        print(f"Model name: {model_name}")
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        mlp_results_path = f"results/proj_experiment/mlp_results/{folder_name}/proj"

        results_dir = f"results/proj_experiment/improve_performance/final_mlp/{folder_name}/mlp"
        create_results_dir(results_dir)

        model = load_model(model_name, cache_dir)
        
        mlp_intervention(model, mlp_results_path, data_dir, results_dir, n_paren=n_paren, metric="f1-score")
        # mlp_intervention(model, mlp_results_path, data_dir, results_dir, n_paren=n_paren, metric="precision")

        # mlp_intervention(model, mlp_results_path, data_dir, results_dir, n_paren=n_paren, metric="recall")

        del model
        clear_cache()


if __name__ == "__main__":
    main()