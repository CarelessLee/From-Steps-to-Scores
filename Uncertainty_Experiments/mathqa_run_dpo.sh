#!/bin/bash

# Define default values for the arguments
model_name_or_path="meta-llama/Meta-Llama-3-8B"
dataset_path="DPO_Data/fully_processed_mathqa_responses.json"
output_dir="mathqa_dpo_training_output"
deepspeed_config="ds_config_zero3_dpo.json"
deepspeed_args="--master_port=12231"

# Parse command line arguments to override defaults
while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_dir)
      output_dir="$2"
      shift
      ;;
    --deepspeed_config)
      deepspeed_config="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

exp_id=dpo_finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

# Run the DPO training
deepspeed --include localhost:0,1,2,3 ${deepspeed_args} \
  new_dpo.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    --deepspeed_config ${deepspeed_config} \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
