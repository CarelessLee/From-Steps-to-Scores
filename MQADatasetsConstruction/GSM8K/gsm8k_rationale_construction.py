import openai
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser
import os

# Initialize OpenAI GPT-4o
openai.api_key = '' 

def call_gpt4o(problem):
    prompt = (f"Problem: {problem}\n"
              "Please provide a succinct step by step solution for the question above in the following format, without any extra wording:\n"
              "Step 1: (logical step 1)\n"
              "Step 2: (logitcal step 2)\n"
              "...\n"
              "Step n: (logitcal last step)\n"
              "Result: (Final result)\n"
              "Please strictly stick to the format above")

    response = openai.ChatCompletion.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You are an expert in solving math problems."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.8
    )
    rationale = response.choices[0].message['content'].strip()
    return rationale

def main():
    results = []
    # Made changes from train -> test for testset generation
    with open("data/processed_test.jsonl", 'r') as f:
        for i, line in enumerate(tqdm(f, total=10)):
            if i >= 10:
                break
            sample = json.loads(line)
            problem = sample["question"]
            correct_answer = sample["answer"]
            
            rationale = call_gpt4o(problem)

            print('rationale: ', rationale)
            
            result_entry = {
                "Problem": problem,
                "Rationale": rationale,
                "correct": correct_answer
            }
            results.append(result_entry)

    os.makedirs("results", exist_ok=True)
    with open("results/gsm8k_test_rationale_raw_1_to_10.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
