import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import tempfile
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load CodeLlama-7B model and tokenizer
model_name = "codellama/CodeLlama-7b-hf"  # Replace with your specific model variant
cache_dir = "../../../projects/ziyuyao/codellama/codellama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)

# Load HumanEval dataset
dataset = load_dataset("openai_humaneval", split="test")

def evaluate_humaneval():
    correct = 0
    total = len(dataset)
    
    for problem in dataset:
        # Prepare prompt and test cases
        prompt = problem["prompt"]
        test_code = problem["test"]
        
        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0  # Greedy decoding for pass@1
        )
        
        # Decode and format code
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_code = generated_code + "\n" + test_code
        
        # Test in isolated environment
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
            f.write(full_code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    correct += 1
            except:
                pass
                
    return correct / total * 100

# Run evaluation
accuracy = evaluate_humaneval()
print(f"HumanEval Accuracy: {accuracy:.2f}%")