# This is the Python Implementation for use LLaMA based reward model scoring sample rationales
# and select the best sample for later SFT

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import multiprocessing
from tqdm import tqdm
import argparse
import json
import time
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_name_or_path", type=str, default='peiyi9979/math-shepherd-mistral-7b-prm')  # model path
    parser.add_argument("--dataset", type=str, default='rso_data')  # data path
    parser.add_argument("--output_dir", type=str, default="LMFlow")  # output dir
    parser.add_argument("--num_gpus", type=int, default=8)  # output dir
    parser.add_argument("--local_rank", type=int, default=0)  
    return parser.parse_args()

def batch_data(data_list, batch_size=8):
    n = batch_size
    batch_data = []
    for i in range(n-1):
        start = i * (len(data_list) // batch_size)
        end = (i+1)* (len(data_list) // batch_size)
        batch_data.append(data_list[start:end])

    last_start = (n-1) * (len(data_list) // batch_size)
    batch_data.append(data_list[last_start:len(data_list)])
    return batch_data

def select_sample(args,sample,model,tokenizer,candidate_tokens,step_tag_id):
    prompt = sample['prompt']
    scores_list = []
    scores_save = []
    sample_list = [prompt + sample['answers'][i] for i in range(len(sample['answers']))]
    input_id = torch.tensor(tokenizer(sample_list,padding=True)['input_ids']).to(args.local_rank)
    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,0] 
        for i in range(scores.shape[0]):
            current = scores[i]
            step_scores = current[input_id[i] == step_tag_id]
            if len(step_scores):
                scores_list.append(min(step_scores))
                scores_save.append(step_scores)
            else:
                scores_list.append(0)
        idx = scores_list.index(max(scores_list))
        text = f"{prompt} {sample['answers'][idx]}"
        return text, scores_save


def predict(args, text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(args.local_rank)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    certainty_score = logits.item()
    return certainty_score


def format_prompt_and_select(args, sample, model, tokenizer):
    # sample_list = [f"Problem: {sample['question']}\n---\nRationale Step: {rationale}" for rationale in sample['sampled_rationales']]
    highest_score = -1
    scores_list = []
    best_rationale = ""
    for rationale in sample["sampled_rationales"]:
        text = f"Problem: {sample['question']}\n---\nRationale Step: {rationale}"
        predicted_certainty_score = predict(args, text, model, tokenizer)
        scores_list.append(predicted_certainty_score)
        if highest_score < predicted_certainty_score:
            highest_score = predicted_certainty_score
            best_rationale = rationale
    best_rationale = f"{sample['question']} {best_rationale}"
    return best_rationale, scores_list


def worker(args, model, tokenizer, data):

    temp_instances = []
    scores = []

    for sample in tqdm(data):
        text, scores_save = format_prompt_and_select(args, sample, model, tokenizer)
        temp_instances.append({"text":text})
        scores.append(scores_save)
    return temp_instances, scores

    # if "Llama-3" in args.reward_name_or_path:
    #     print("not implemented")
    # else:
    #     good_token = '+'
    #     bad_token = '-'
    #     step_tag = 'ки'
    #     candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
    #     step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
    # try:
    #     for sample in tqdm(data):
    #         text, scores_save = select_sample(args,sample,model,tokenizer,candidate_tokens,step_tag_id)
    #         temp_instances.append({"text":text})
    #         scores.append(scores_save)
        
    #     # Save results
    #     return temp_instances,scores
    # except:
    #     return temp_instances,scores




if __name__ == "__main__":
    args = parse_args()

    print("---------------")
    print("begin to load reward model.")
    print("---------------")

    downloaded = False
    while not downloaded:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
            model = AutoModelForSequenceClassification.from_pretrained(args.reward_name_or_path, use_flash_attention_2=True,torch_dtype=torch.bfloat16).to(args.local_rank).eval()
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            print("Failed to load the reward model. Retrying....")
            time.sleep(2)

    #pading left or padding right???
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("---------------")
    print("begin to load the sampling data")
    print("---------------")
    
    with open(f"{args.dataset}/samples_{args.local_rank}.json",'r') as f:
        data = json.load(f)
    
    print("---------------")
    print("begin evaluation")
    print("---------------")
    
    eval_data,scores = worker(args, model, tokenizer, data)
    
    print("---------------")
    print("Successfully filter the training data from the reward model")
    print("Now begin to save the data")
    print("---------------")     
    with open(f"{args.output_dir}/result_{args.local_rank}.json",'w') as f:
        json.dump(eval_data,f,indent=4,ensure_ascii=False)
    
    # with open(f"reward/scores_{args.local_rank}.json",'w') as f:
    #     json.dump(scores,f,indent=4,ensure_ascii=False)
    print("---------------")
    print("Saved the filtered data successfully!")
    print("---------------")
