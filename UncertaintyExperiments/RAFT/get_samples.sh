# This is the entrace bash file to use policy model to sample step wise ratioanles

python rso_generate_llama.py  \
    --model_name_or_path $1 \
    --current_iter $2 \
    --dataset openai/gsm8k \
    --batch_size 256 \
    --batch_size_per_iter 1000 \
    --tensor_parallel_size 1 \
    --num_gpus $3 \
    --output_dir $4 \
    --sanity_check $5 \
    --local_rank $6 \
    --random_seed $7 
