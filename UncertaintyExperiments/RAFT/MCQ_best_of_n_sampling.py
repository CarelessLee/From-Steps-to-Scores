import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

with open('MCQTestset/GSM8K_Full_Test.json', 'r') as file:
    data = json.load(file)

subset_data = data

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    revision="main",
    torchscript=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to get step-by-step rationale from llama3-8b-Instruct
def get_rationale(question, n_samples=10, temperature=0.9):
    prompt = (
        f"Problem: {question}\n"
        "Please provide a succinct step-by-step solution for the question above in the following format, without any extra wording:\n"
        "[START]\n"
        "Step 1: (logical step 1)\n"
        "Step 2: (logical step 2)\n"
        "...\n"
        "Step n: (logical last step)\n"
        "Result: (Final result)\n"
        "[END]\n"
        "Please strictly stick to the format above."
    )
    
    sampled_rationales = []
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    for _ in range(n_samples):
        outputs = model.generate(
            input_ids,
            max_new_tokens=350,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        rationale = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("rationale: ", rationale)
        sampled_rationales.append(rationale)
    
    return sampled_rationales

# Generate rationales for each question
result_data = []
for item in tqdm(subset_data, desc="Processing questions"):
    question = item['question']
    answer = item['answer']
    rationales = get_rationale(question)
    result_data.append({
        "question": question,
        "answer": answer,
        "sampled_rationales": rationales
    })

with open('MCQTestset/MCQ_llama3_instruct_test.json', 'w') as outfile:
    json.dump(result_data, outfile, indent=4)
