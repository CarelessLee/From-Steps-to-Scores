deepspeed_args="--master_port=11100"
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
# CUDA_DEVICES=$CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# deepspeed --include localhost:${CUDA_DEVICES} ${deepspeed_args} \
log_dir="./sft_log"
mkdir -p ${log_dir}
#deepspeed ${deepspeed_args} \
deepspeed --include localhost:0,1,2,3,4,5 ${deepspeed_args} \
  sft.py \
    --model_name_or_path $1 \
    --dataset_path $3 \
    --output_dir $2 --overwrite_output_dir \
    --current_iteration $4 \
    --total_iteration $5 \
    --random_seed $6 \
    --num_train_epochs 2 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_step 1 \
    --gradient_checkpointing True \
    --use_flash_attention True \
    --deepspeed configs/ds_config_zero2.json \
    --bf16 True \
    --run_name metamath_iter_$4 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err

export CUDA_VISIBLE_DEVICES=CUDA_DEVICES
