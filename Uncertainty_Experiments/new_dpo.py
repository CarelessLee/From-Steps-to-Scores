import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import deepspeed

# Function to transform the dataset example based on certainty_score
def transform_example_for_dpo(example):
    if example['certainty_score'] >= 0.5:
        chosen = example['model_response']
        rejected = "I don't know"
    else:
        chosen = example['original_answer']
        rejected = example['model_response']
    
    return {
        'prompt': example['question'],
        'chosen': chosen,
        'rejected': rejected,
        'certainty_score': example.get('certainty_score', 0.0) 
    }

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='DPO Training with TRL and DeepSpeed')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to pre-trained model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the trained model')
    parser.add_argument('--deepspeed_config', type=str, required=True, help='Path to DeepSpeed configuration file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()

    deepspeed.init_distributed() 
    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to("cpu")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_dataset = load_dataset('json', data_files=args.dataset_path)
    train_dataset = raw_dataset['train']
    
    # Transform the dataset for DPO
    train_dataset = train_dataset.map(transform_example_for_dpo, num_proc=4)
    print("Transformed dataset structure for DPO:")
    print(train_dataset[0])


    print(f"Transformed training dataset size: {len(train_dataset)}")
    for i in range(3):
        print(train_dataset[i])

    dpo_config = DPOConfig(
        remove_unused_columns=False,  # Keep all columns to avoid 'out of index' issues
        beta=0.1,  
        output_dir=args.output_dir,
        learning_rate=1e-5,  
        per_device_train_batch_size=2,  
        gradient_accumulation_steps=4,  
        num_train_epochs=2,  
        logging_steps=10,  
        local_rank=args.local_rank, 
        deepspeed=args.deepspeed_config 
    )

   
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=train_dataset 
    )

    try:
        dpo_trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")

    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
