from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
from tqdm import tqdm  # Import tqdm for progress bars

# Load the trained reward model and tokenizer
model_name_or_path = '/home/jzhanggr/jzhanggr001/Uncertainty_Experiments/mathqa_model'  # Path to your reward model
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    # Perform inference with the model
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the logits and compute the certainty score
    logits = outputs.logits
    certainty_score = torch.sigmoid(logits).item()  # Convert logits to a probability score
    return certainty_score

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
    Process the input JSON file to create a new JSON file with refined responses and certainty scores.
    
    Args:
    - input_file: Path to the input JSON file (e.g., 'gsm8k_responses.json').
    - output_file: Path to the output JSON file (e.g., 'processed_gsm8k_responses.json').
    """
    # Load the dataset
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    processed_data = []
    
    # Wrap the processing loop in tqdm to show progress
    for entry in tqdm(data, desc="Processing Responses", unit="entry"):
        # Refine the model response to keep only the relevant steps
        refined_response = refine_response(entry['model_response'])
        
        # Compute the certainty score using the reward model
        certainty_score = predict(refined_response)
        
        # Add the score to the entry
        processed_entry = {
            "question": entry["question"],
            "original_answer": entry["original_answer"],
            "model_response": refined_response,
            "certainty_score": certainty_score  # Add the certainty score to the entry
        }
        
        processed_data.append(processed_entry)
    
    # Save the processed data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)
    
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # Define the input and output file paths
    input_file = 'mathqa_responses.json'  # The input file containing the original data
    output_file = 'fully_processed_mathqa_responses.json'  # The output file for saving processed data
    
    # Process the responses and add the certainty scores
    process_responses(input_file, output_file)
