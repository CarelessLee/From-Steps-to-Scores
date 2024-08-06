prompt = (f"Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
"Please provide a succinct step-by-step solution for the question above in the following format, without any extra wording:\n"
"[START]\n"
"Step 1: (logical step 1)\n"
"Step 2: (logical step 2)\n"
"...\n"
"Step n: (logical last step)\n"
"Result: (Final result)\n"
"[END]\n"
"Please strictly stick to the format above.")


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
        return "Problem statement not found."
    
print(extract_math_problem(prompt))