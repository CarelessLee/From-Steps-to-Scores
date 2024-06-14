import openai
import json
from tqdm.auto import tqdm
import os
import numpy as np

# Initialize OpenAI GPT-4
openai.api_key = ''
k = 10

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def call_gpt4_o(engine, problem, rationale_step):
    prompt = (f"Problem: {problem}\n"
              f"Rationale so far:\n{rationale_step}\n"
              "Please answer the probem based on the Rationale. Only return a numerical value for the answer.")

    # Sampling model's response for k times
    answers = []

    for _ in range(k):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in solving math problems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8,
                temperature=0.8,
                logprobs=True,
                top_logprobs=20
            )
            answer = response.choices[0].message['content'].strip()
            # print("answer: ", answer)

            if response.choices[0].logprobs and response.choices[0].logprobs.content[0]:
                token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                logits = {logprob['token']: logprob['logprob'] for logprob in token_logprobs}
                tokens = list(logits.keys())
                logit_values = np.array(list(logits.values()))
                probabilities = softmax(logit_values)
                token_probs = dict(zip(tokens, probabilities))
            else:
                token_probs = {}

            answers.append({'answer': answer, 'logits': token_probs})
            # print("coverted_logits: ", token_probs)

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            answers.append({"answer": None, "probabilities": {}})

    return answers
    # print("token_logprobs: ", token_logprobs)
    # logits = {token: token_logprobs[token] for token in ['A', 'B', 'C', 'D', 'E'] if token in token_logprobs}
    # print("logits: ", logits)


def split_rationale_steps(rationale):
    if "Result:" in rationale:
        rationale = rationale.split("Result:")[0]
    
    steps = rationale.split('Step ')[1:]  # Split on 'Step ' and ignore the first empty part
    steps = ['Step ' + step.strip() for step in steps]  # Prepend 'Step '
    
    return steps

def main():
    with open("results/gsm8k_test_rationale_raw_1_to_10.json", 'r') as f:
        data = json.load(f)

    results = []
    counter = 1
    for sample in tqdm(data):
        problem = sample["Problem"]
        rationale = sample["Rationale"]
        
        steps = split_rationale_steps(rationale)
        stepwise_results = []
        
        for i in range(1, len(steps) + 1):
            rationale_step = ' '.join(steps[:i])
            sampled_answers = call_gpt4_o(openai, problem, rationale_step)
            
            stepwise_result = {
                "step": i,
                "rationale_step": rationale_step,
                "sampled_answers": sampled_answers
            }

            # print("step: ", i)
            # print("rational_step: ", rationale_step)
            # print("answer: ", answer)
            # print("logits: ", logits)

            stepwise_results.append(stepwise_result)
        
        result_entry = {
            "PID": counter,
            "Problem": problem,
            "Rationale": rationale,
            "correct": sample["correct"],
            "stepwise_results": stepwise_results
        }
        counter += 1
        print("result_entry: ", result_entry)
        results.append(result_entry)


    os.makedirs("results", exist_ok=True)
    with open("results/20240607_sampled_test_responses.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
