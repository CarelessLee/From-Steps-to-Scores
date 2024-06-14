model_name_or_path=meta-llama/Meta-Llama-3-8B
dataset_path=deepmind_train.json
output_dir=deepmind_model
deepspeed_args="--master_port=11222"

exp_id=pred
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/gsm8k_log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed --include localhost:2,3 ${deepspeed_args} \
  prediction_model_train.py \
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
