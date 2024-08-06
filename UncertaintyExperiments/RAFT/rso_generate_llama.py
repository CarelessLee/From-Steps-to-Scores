import argparse
import json
import re
import torch
import stanza
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
import time
import random

def format_dataset(raw_datasets):
    formatted_dataset = []
    for sample in raw_datasets:
        prompt = (
            f"Problem: {sample['question']}\n"
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
        formatted_dataset.append(prompt)
    return formatted_dataset

def format_half_dataset(raw_datasets):
    gsm_prompt = []
    math_prompt = []
    for sample in raw_datasets:
        if "GSM" in sample['type']:
            gsm_prompt.append(sample['query'])
        else:
            math_prompt.append(sample['query'])
    return gsm_prompt, math_prompt

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    batch_data.append(data_list[last_start:len(data_list)])
    return batch_data

def process_answer(args,answer_list):
    post_list = []
    for sample in answer_list:
        doc = snlp(sample)
        doc_sents = [sentence.text for sentence in doc.sentences]
        
        truncate_doc = []
        for i in doc_sents:
            if "The answer is" in i:
                truncate_doc.append(i) 
                break
            else:
                truncate_doc.append(i)
        
        if "Llama-3" in args.model_name_or_path:
            post_list.append(" ".join(truncate_doc))
        else:
            temp = " ".join(truncate_doc)
            temp = temp.replace(" ки "," ки\n")
            post_list.append(temp)
    return post_list

def save(store_data,args):
    with open(f"{args.output_dir}/samples_{args.local_rank}.json",'w') as f:
        json.dump(store_data,f,indent=4,ensure_ascii=False)


def extract_content(text):
    # Define the regex pattern to find content between [START] and [END]
    pattern = re.compile(r'\[START\](.*?)\[END\]', re.DOTALL)
    # Search for the pattern in the provided text
    match = pattern.search(text)
    if match:
        # Return the matched content, stripping leading and trailing whitespace
        return match.group(1).strip()
    else:
        # Return an empty string if no match is found
        return ""


def extract_math_problem(prompt):
    start_marker = "Problem: "
    end_marker = "Please provide a succinct step-by-step solution"
    
    # Find the start and end positions of the problem statement
    start_index = prompt.find(start_marker) + len(start_marker)
    end_index = prompt.find(end_marker)
    
    # Extract the problem statement
    if start_index != -1 and end_index != -1:
        problem_statement = prompt[start_index:end_index].strip()
        return problem_statement
    else:
        return ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')  # model path
    parser.add_argument("--dataset", type=str, default='openai/gsm8k')  # data path
    parser.add_argument("--batch_size", type=int, default=1024)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=2)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="rso_data")  # output location
    parser.add_argument("--current_iter", type=int, default=0)  # current iteration
    parser.add_argument("--num_gpus",type=int,default=2)
    parser.add_argument("--local_rank",type=int,default=0)
    parser.add_argument("--batch_size_per_iter", type=int, default=4096)  # number of samples per iteration
    parser.add_argument("--sanity_check", type=int, default=0)  # sanity check
    parser.add_argument("--random_seed", type=int, default=42)  # random seed
    parser.add_argument("--sample_n", type=int, default=10) # number of sampling times for each prompt
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    random.seed(args.random_seed)
    
    raw_datasets = load_dataset(args.dataset)
    raw_datasets = raw_datasets['train']
    if args.sanity_check:
        prompt = format_dataset(raw_datasets)[args.current_iter*20:(args.current_iter+1)*20]
    else:
        #original version
        all_prompt = format_dataset(raw_datasets)
        prompt = all_prompt[args.current_iter*args.batch_size_per_iter:(args.current_iter+1)*args.batch_size_per_iter]
        prompt = prompt[int(len(prompt)*args.local_rank/args.num_gpus):int(len(prompt)*(args.local_rank+1)/args.num_gpus)]

    random.shuffle(prompt)

    batch_prompt = batch_data(prompt, batch_size=args.batch_size)
    stop_tokens = []
    sampling_params = SamplingParams(n=args.sample_n, temperature=0.9, top_p=0.9, max_tokens=512, stop=stop_tokens, seed=args.random_seed)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, dtype = "float16",enforce_eager=True, gpu_memory_utilization=0.95,swap_space=32)

    print("---------------")
    print("begin to sampling from the SFT model")
    print("---------------")
    
    store_data = []
    count = 0
    for idx, prompt in tqdm(enumerate(batch_prompt), total=len(batch_prompt)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            cleaned_rationales = []
            prompt = output.prompt
            generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
            
            # extract the math problem
            question = extract_math_problem(prompt)
            if question == "":
                continue

            # process and extract the rationale
            for text in generated_text:
                extracted_rationale = extract_content(text)
                if extracted_rationale != "" and extracted_rationale.startswith('Step 1:'):
                    cleaned_rationales.append(extracted_rationale)
                    #f"Problem: {sample['question']}\n---\nRationale Step: {rationale}"

            store_data.append({"question":question, "sampled_rationales":cleaned_rationales})
        count += 1
        if count % 1 == 0:
            save(store_data, args)
    
    print("---------------")
    print("Successfully sampled from the SFT model")
    print("Now begin to save the data")
    print("---------------")       
    save(store_data,args)
    print("---------------")
    print("Saved the sampling data successfully!")
    print("---------------")
