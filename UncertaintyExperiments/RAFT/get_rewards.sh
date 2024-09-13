# This is the entrance bash file to use reward model to score the sampled rationales, and select the best sample based on scores
# use rso_reward_llama.py for llama based reward model
# use rso_reward.py for mistral based reward model, i.e. Math-Shepherd PRM

python rso_reward_llama.py \
    --dataset $1 \
    --output_dir $2 \
    --reward_name_or_path $3 \
    --num_gpus $4 \
    --local_rank $5
    