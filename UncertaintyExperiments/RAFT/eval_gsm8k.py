import argparse
import json
import re
import torch
from tqdm import tqdm
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
import time
MAX_INT = sys.maxsize

def extract_math_shepherd_results(text):
    sen_pattern = r'The answer is: [^.]+\.'
    # Find all matches
    sen_matches = re.findall(sen_pattern, text)

    # Regular expression to match numbers with optional commas
    num_pattern = r'\b\d{1,3}(?:,\d{3})*\b|\b\d+\b'

    all_num_matches = []
    for sentence in sen_matches:
        # Find all matches
        num_matches = re.findall(num_pattern, sentence)
        all_num_matches += num_matches
    
    # Remove commas from numbers like '150,000' and convert to integers
    numbers = [int(match.replace(',', '')) for match in all_num_matches]
    
    return numbers


def extract_results_with_commas(llm_output):
    # Use regex to find all occurrences of numbers (with or without commas) after "Result: "
    results = re.findall(r'Result:\s*\$?([\d,]+)', llm_output)

    # Remove commas from the numbers and convert them to integers
    results_cleaned = [int(result.replace(',', '')) for result in results]

    return results_cleaned

def extract_result(rationale):
    # example rationale: 'Step 1: ... Step 2: ... Step n: Result: 666'
    # Search for the final numerical value after 'Result:'
    match_after_result = re.search(r'Result:\s*(.*?)(?:\.\s*|$)', rationale)
    if match_after_result:
        # Extract the last number from the matched result text
        final_numbers = re.findall(r'([\d,]+(?:\.\d+)?)', match_after_result.group(1))
        if final_numbers:
            return final_numbers[-1].strip()

    # Fallback: Search for the last number before 'Result:'
    result_index = rationale.find('Result:')
    if result_index != -1:
        matches_before_result = re.findall(r'([\d,]+(?:\.\d+)?)', rationale[:result_index])
        if matches_before_result:
            return matches_before_result[-1].strip()

    return None

def extract_result_all(completion, gt):
    y_pred = extract_answer_number(completion)
    if y_pred is not None:
        y_pred = str(y_pred)
        y_pred = y_pred.replace(",", "")
    if y_pred not in (None, "") and float(y_pred) == float(gt):
        return y_pred, True

    y_pred = get_last_answer(completion)
    if y_pred is not None:
        y_pred = str(y_pred)
        y_pred = y_pred.replace(",", "")
    if y_pred not in (None, "") and float(y_pred) == float(gt):
        return y_pred, True
    
    y_pred = extract_result(completion)
    if y_pred is not None:
        y_pred = str(y_pred)
        y_pred = y_pred.replace(",", "")
    if y_pred not in (None, "") and float(y_pred) == float(gt):
        return y_pred, True

    y_pred_list = extract_results_with_commas(completion)
    if len(y_pred_list) > 0:
        for y_pred in y_pred_list:
            if y_pred is not None:
                y_pred = str(y_pred)
                y_pred = y_pred.replace(",", "")
            if y_pred not in (None, "") and float(y_pred) == float(gt):
                return y_pred, True

    y_pred_list = extract_math_shepherd_results(completion)
    if len(y_pred_list) > 0:
        for y_pred in y_pred_list:
            if y_pred not in (None, "") and float(y_pred) == float(gt):
                return y_pred, True

    return y_pred, False



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def get_last_answer(completion):
    answer = ""
    temp = completion
    temp = temp.replace(",", "")
    temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    if len(temp) != 0:
        answer = temp[-1]
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]
            # round the answer to thenearest integer
            try:
                answer = str(round(float(answer)))
            except:
                answer = answer[:-1]
    if answer == "":
        return None
    return answer


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            #temp_instr = problem_prompt.format(instruction=item["query"])
            temp_instr = item['query']
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=700, stop=stop_tokens)
    print('sampleing =====', sampling_params)

    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, enforce_eager=True, dtype = "float16", gpu_memory_utilization=0.9)
            
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in tqdm(enumerate(zip(batch_gsm8k_ins, gsm8k_answers))):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        #y_pred = extract_answer_number(completion)
        #if y_pred is None:
            #y_pred = get_last_answer(completion)
        #if y_pred != None:
            #if float(y_pred) == float(prompt_answer):
                #result.append(float(y_pred) == float(prompt_answer))
            #elif str(extract_result(completion)) == str(prompt_answer):
                #result.append(True)
        y_pred, success = extract_result_all(completion, prompt_answer)
        if success:
            result.append(True)
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'pred':y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
        #else:
            #result.append(False)
            #temp = {'question': prompt, 'output': completion, 'pred':y_pred, 'answer': prompt_answer}
            #invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    #print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    #print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    output_dir = args.output_dir.replace("/","_")
    with open(f"eval_result/{output_dir}",'w') as f:
        json.dump({"gsm8k":acc},f)
        json.dump(invalid_outputs, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = 'iterative_rft/model3')  # model path
    parser.add_argument("--data_file", type=str, default='data/test/GSM8K_test.jsonl')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=1000)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=2)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    print("---------------")
    print("begin to evaluate the gsm8k dataset.")
    print("---------------")
    gsm8k_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
