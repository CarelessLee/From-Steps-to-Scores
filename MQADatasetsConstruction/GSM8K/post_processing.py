import json
from sklearn.metrics import average_precision_score

file_path = 'results/20240607_sampled_test_responses.json'
with open(file_path, 'r') as file:
    data = json.load(file)

processed_data = []
for question in data:
    correct_answer = question['correct']
    new_question = {
        "PID": question["PID"],
        "Problem": question["Problem"],
        "Rationale": question["Rationale"],
        "correct": correct_answer,
        "stepwise_results": []
    }
    
    # Process each Rationale step
    for step in question["stepwise_results"]:
        sampled_answers = step["sampled_answers"]
        step_correctness_list = []
        step_logits_list = []
        ap_score = 0
        
        # Process each sampled response
        for response in sampled_answers: 
            given_answer = response["answer"]
            # print("model answer: ", given_answer)
            logits = response.get("logits", {})
            
            # Check if the answer is correct
            is_correct = int(given_answer is not None and given_answer.lower() == correct_answer.lower())
            step_correctness_list.append(is_correct)
            
            # Get the logit for the correct answer
            upper_logit = logits.get(correct_answer.upper(), 0)
            lower_logit = logits.get(correct_answer.lower(), 0)
            correct_logit = upper_logit + lower_logit if upper_logit and lower_logit else upper_logit or lower_logit
            step_logits_list.append(correct_logit)
        
        if step_logits_list and step_correctness_list:
        	ap_score = average_precision_score(step_correctness_list, step_logits_list)

        new_step = {
            "step": step["step"],
            "rationale_step": step["rationale_step"],
            # "correctness_list": step_correctness_list,
            # "logits_list": step_logits_list,
            "certainty_score": ap_score
        }
        new_question["stepwise_results"].append(new_step)
    
    processed_data.append(new_question)

# Save the processed data to a new JSON file
output_file_path = 'results/20240607_processed_test_responses_1_to_10.json'
with open(output_file_path, 'w') as outfile:
    json.dump(processed_data, outfile, indent=4)
