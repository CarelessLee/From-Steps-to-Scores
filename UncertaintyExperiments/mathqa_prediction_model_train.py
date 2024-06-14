import json
import random
import torch
import numpy as np
import evaluate
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, Dataset
from transformers import (
    TrainingArguments, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer,
    HfArgumentParser,
    DataCollatorWithPadding
)
    
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default="", metadata={"help": "the model path"})
    dataset_path: str = field(default="", metadata={"help": "dataset path"})
    validation_split_percentage: int = field(default=10,metadata={"help":"validation percentage"})

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
  
if __name__ == "__main__":

    parser = HfArgumentParser((TrainingArguments,ScriptArguments))
    training_args,script_args = parser.parse_args_into_dataclasses()
    
    model = AutoModelForSequenceClassification.from_pretrained(script_args.model_name_or_path, 
                                                               num_labels=1,
                                                                torch_dtype=torch.bfloat16,
                                                                use_cache=False,
                                                                use_flash_attention_2=True)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path,use_fast=True)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "right"
    
    with open(script_args.dataset_path,'r') as f:
        label_data = json.load(f)
    
    data = {
        "text":[],
        "label":[]
    } 
    for sample in label_data:
        for step in sample['stepwise_results']:         
            data['text'].append(
                f"Problem: {sample['Problem']}\n"
                f"---\n"
                f"Options: {sample['options']}\n"
                f"---\n"
                f"Rationale Step: {step['rationale_step']}"
            )
            # data['text'].append(
            #     f"### Problem ###\n{sample['Problem']}\n\n"
            #     f"### Options ###\n{sample['options']}\n\n"
            #     f"### Task ###\n"
            #     f"Please evaluate the certainty score for the following rationale step on a scale from 0 to 1. "
            #     f"Ensure that your evaluation is based on:\n"
            #     f"- The problem stated above\n"
            #     f"- The given options\n\n"
            #     f"Rationale Step:\n{step['rationale_step']}"
            # )

            data['label'].append(step['certainty_score'])
               
    dataset_object = Dataset.from_dict(data)
    dataset_object = dataset_object.shuffle()
    
    eval_dataset = None
    if script_args.validation_split_percentage > 0:
        idx_gap = int((1-script_args.validation_split_percentage/100) * len(dataset_object))
        train_dataset = dataset_object.select(range(idx_gap))
        eval_dataset = dataset_object.select(range(idx_gap, len(dataset_object)))
    else:
        train_dataset = dataset_object
        
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(preprocess_function, batched=True)
        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)