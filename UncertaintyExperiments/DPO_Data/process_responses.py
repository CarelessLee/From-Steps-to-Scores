import json

def refine_response(response):
    """
    Function to refine the model response by extracting only the relevant step-by-step rationale.
    Modify this based on the typical structure of the responses.
    """
    # Start the extraction from the first "Step 1:" found in the response
    start_index = response.find("Step 1:")
    if start_index == -1:
        # If "Step 1:" is not found, return the entire response (indicating no structured steps found)
        return response
    
    # Extract only the steps and rationale
    steps_response = response[start_index:]
    return steps_response

def process_responses(input_file, output_file):
    """
    Process the input JSON file to create a new JSON file with refined responses.
    
    Args:
    - input_file: Path to the input JSON file (e.g., 'gsm8k_responses.json').
    - output_file: Path to the output JSON file (e.g., 'processed_gsm8k_responses.json').
    """
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    processed_data = []
    
    for entry in data:
        # Retain the original question and original answer
        processed_entry = {
            "question": entry["question"],
            "original_answer": entry["original_answer"],
            "model_response": refine_response(entry["model_response"])
        }
        processed_data.append(processed_entry)
    
    # Save the processed data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)
    
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # Define the input and output file paths
    input_file = 'gsm8k_responses.json'
    output_file = 'processed_gsm8k_responses.json'
    
    # Process the responses
    process_responses(input_file, output_file)
