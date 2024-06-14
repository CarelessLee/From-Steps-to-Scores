import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm  # Import tqdm for progress bars

# Specify the GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate responses with step-by-step reasoning
def generate_response(question, model, tokenizer, device):
    prompt = (
        f"Question: {question}\n"
        "Please solve this problem step-by-step. Explain each step clearly and provide the final answer at the end.\n"
        "Step 1: Identify and describe the problem.\n"
        "Step 2: Outline the approach to solve the problem.\n"
        "Step 3: Solve the problem step by step.\n"
        "Step 4: Provide the final answer.\n"
        "Let's start with Step 1."
    )
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=1024, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Function to process the dataset
def process_dataset(dataset, model, tokenizer, output_file, device, include_options=False):
    responses = []
    for idx, item in enumerate(tqdm(dataset[:1000], desc=f"Processing {output_file}")):  # Add progress bar
        question = item.get("question") or item.get("Problem")
        original_answer = item.get("answer") or item.get("correct")
        answer = generate_response(question, model, tokenizer, device)
        
        response_data = {
            "question": question,
            "original_answer": original_answer,
            "model_response": answer
        }

        print("model_response: ", answer)

        if include_options and "options" in item:
            response_data["options"] = item["options"]

        responses.append(response_data)

    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=4)

# Load datasets
with open('deepmind_raw.json') as f:
    deepmind_data = json.load(f)

with open('gsm8k_raw.json') as f:
    gsm8k_data = [json.loads(line) for line in f.readlines()]

with open('mathqa_raw.json') as f:
    mathqa_data = json.load(f)

# Function to assign models to GPUs
def process_on_gpu(dataset, output_file, device, include_options=False):
    # Load a new model instance on the specified device
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    process_dataset(dataset, model, tokenizer, output_file, device, include_options)

# Assign each dataset to a specific GPU
gpu_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
datasets = [
    (deepmind_data, 'deepmind_responses.json', False),
    (gsm8k_data, 'gsm8k_responses.json', False),
    (mathqa_data, 'mathqa_responses.json', True)  # Include options for mathqa
]

# Process each dataset on a separate GPU using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:
    for idx, (dataset, output_file, include_options) in enumerate(datasets):
        executor.submit(process_on_gpu, dataset, output_file, gpu_devices[idx], include_options)

print("Processing started. Check the output JSON files after completion.")
