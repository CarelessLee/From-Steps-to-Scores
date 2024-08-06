import json
import re
import argparse

# def extract_second_segment(rationale):
#     # Find all occurrences of the segments between [START] and [END]
#     segments = re.findall(r'\[START\](.*?)\[END\]', rationale, re.DOTALL)
#     # Return the second segment if it exists
#     print(segments)
#     print(len(segments))
#     if len(segments) >= 2:
#         print("HERE")
#         return segments[1].strip()
#     return rationale  # Return the original rationale if the second segment is not found


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


def process_rationales(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Process each item and its sampled rationales
    for item in data:
        cleaned_rationales = []
        for rationale in item['sampled_rationales']:
            extracted_rationale = extract_content(rationale)
            if extracted_rationale and extracted_rationale.startswith('Step 1:'):
                cleaned_rationales.append(extracted_rationale)
        item['sampled_rationales'] = cleaned_rationales
        
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Processed rationales have been saved to {output_file}")

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input_file', type=str)
    # parser.add_argument('output_file', type=str)

    # args = parser.parse_args()


    # process_rationales(args.input_file, args.output_file)
    temp = " \nHere's the solution:\n\n[START]\nStep 1: Natalia sold 48 clips in April.\nStep 2: She sold half as many clips in May. Since she sold 48 clips in April, she sold 48 ÷ 2 = 24 clips in May.\nStep 3: To find the total number of clips she sold, we add the number of clips sold in April and May: 48 + 24 = 72.\nResult: Natalia sold 72 clips altogether in April and May.\n[END]"
    print(extract_content(temp))

if __name__ == '__main__':
    main()
