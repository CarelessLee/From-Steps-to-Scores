model_name_or_path=meta-llama/Meta-Llama-3-8B
dataset_path=mathqa_dataset.json
output_dir=mathqa_model
deepspeed_args="--master_port=13124"

exp_id=pred
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

#testing, rememeber to remove
# export NCCL_DEBUG=INFO
# export TOKENIZERS_PARALLELISM=false

deepspeed --include localhost:0,1 ${deepspeed_args} \
  mathqa_prediction_model_train.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --deepspeed ds_config_zero2.json \
    --bf16 \
    --run_name prediction \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --warmup_ratio 0.03\
    --validation_split_percentage 10 \
    --evaluation_strategy steps\
    --eval_steps 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err

# model_name_or_path=meta-llama/Meta-Llama-3-8B
# dataset_path=gsm8k_dataset.json
# output_dir=gsm8k_model
# deepspeed_args="--master_port=11112"

# exp_id=pred
# project_dir=$(cd "$(dirname $0)"/..; pwd)
# log_dir=${project_dir}/log/${exp_id}
# mkdir -p ${output_dir} ${log_dir}

# deepspeed --include localhost:4,7 ${deepspeed_args} \
#   prediction_model_train.py \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_path ${dataset_path} \
#     --output_dir ${output_dir} --overwrite_output_dir \
#     --num_train_epochs 2 \
#     --learning_rate 1e-5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --deepspeed ds_config_zero2.json \
#     --bf16 \
#     --run_name prediction \
#     --logging_steps 1 \
#     --do_train \
#     --ddp_timeout 72000 \
#     --save_steps 5000 \
#     --warmup_ratio 0.03 \
#     --validation_split_percentage 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1 \
#     --gradient_accumulation_steps 1 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 1 \
#     | tee ${log_dir}/train.log \
#     2> ${log_dir}/train.err
