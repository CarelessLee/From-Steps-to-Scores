# From Steps to Scores

This is the repo for paper 'From Steps to Scores: Improving LLM Certainty and Predicting Uncertainty in Math Reasoning'.



## Introduction

We present a novel framework to quantify and enhance the reliability of large language models (LLMs) in solving complex mathematical problems that require multi-step reasoning. Our approach is structured into **three stages**. First, we introduced “Iterative Stepwise Prompting” (**ISP**), a technique that incrementally reveals and refines reasoning steps to LLMs to improve their certainty. Second, we developed the “Mathematical Confidence Quantifier” (**MCQ**), a Reinforcement Learning-based prediction model to assign certainty scores to each mathematical reasoning step, offering a more granular confidence assessment. Finally, we employed Proximal Policy Optimization (PPO) using MCQ to further enhance the reliability and performance of LLMs. Experiments on the MathQA dataset using Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, and the more advanced GPT-4o models demonstrated considerable gains in accuracy and reduced uncertainty for the ISP phase. This combined methodology provides a robust framework for enhancing LLMs’ performance and reliability in mathematical problem-solving tasks, advancing the development of more trustworthy and capable LLMs.

<img width="561" alt="Screenshot 2024-06-03 at 11 17 16 PM" src="https://github.com/CarelessLee/From-Steps-to-Scores/assets/42497570/22413247-7a78-448c-99f9-4275088fe892">


The repo currently consists of Stage 1 and Stage 2's work for the Certainty Predictor (MCQ) models and PPO/DPO RL models. Directory **MQADatasetsConstruction** consists of scripts that generates training sets for MCQ, and **UncertaintyExperiments** consists of work for training & testing MCQ and training DPO.


## Getting Start

```bash
git clone git@github.com:CarelessLee/From-Steps-to-Scores.git

cd From-Steps-to-Scores
```



## Requirements

The **LMFlow** environment contains all the packages and dependencies required.

```bash
git clone git@github.com:OptimalScale/LMFlow.git
cd LMFLow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh
```



## Processed Datasets

All pre-processed datasets from MathQA, GSM8K, and Mathmatics, and the preliminary reward models trained on these datasets are here: https://drive.google.com/drive/folders/19FbHyj0QOKfcbZSu82radvsALKd5h0e2?usp=sharing



## MCQ 

With the three pre-processed datasets, we can train the MCQ models individually. We provide an example of using **MathQA** dataset with **meta-llama/Meta-Llama-3-8B** model. Other MCQ models could be trained similarly with a change of testset and baseline model.

```bash
# put MathQA json file in the UncertaintyExperiments directory and config 'dataset_path' in mathqa_run_prediction.sh accordingly
cd UncertaintyExperiments
./mathqa_run_prediction.sh
```



This will create a directory 'mathqa_model' that stores the parameters of the trained MCQ model. You can testout MCQ's response using **mathqa_prediction_model_train.py**



## DPO

with the trained MCQ, we can now reinforce LLMs with the help of RLHF algorithms. Here we demonstrate the usecase of DPO. Remember to load the pre-processed datasets (e.g. fully_processed_deepmind_responses.json) into the **UncertaintyExperiments** directory beforehand.

```bash
./new_run_dpo.sh
```

## RAFT
with the trained MCQ, we can now reinforce LLMs with the help of RAFT algorithm.

### Overall Flow of Pipeline
Notice: This script implement multi-process running by paralle running multiple bash script, therefore will generate multiple files.

1. First open the restart_raft.sh and config some important parameters (Details see next section).

2. This script contains several core scripts for different tasks, which are "get_samples.sh", "get_reward.sh", "merge_data.py", "sft.sh", "evaluate.sh".

3. The entire flow starts with get_samples.sh, it will use the policy model to sample step by step rationales to the given dataset. The dataset will pull from hugging face with the given name. The sampled rationales will be output to the infer_set dir of current iteration dir.

4. Second step is get_reward.sh, it will use the reward model specified to score the sampled rationales and select the best one for later SFT. It's input comes from the infer_set of current dir, and output the selected rationales to filter_set of current dir"

5. Third step is the merge_data.py, it simply merges the output files from different GPU to a single file for later SFT.

6. Fourth step is the sft.sh, it will take the selected rationales from filter_set of current iteration dir as training data. The fine-tuned model after training will be saved into next iteration's work dir. i.e. if current work dir is model0, the fine-tuned model will be saved to model1.

7. Fifth step is evaluate.sh, it will test the fine-tuned model on gsm8k and math datasets.

8. The algorithm will repeat the above steps in new iterations until reach the specified number of iteration.



### Important Notes for using the restart_raft.sh script
Use restart_raft.sh, and config some important parameters

1. Please config the parameters as you need, especially the num_gpus and corresponding gpu_list

2. The numeric number at the end of each shell command is the local rank of GPU, make sure it has to be consecutive order
    ex: gpu_list=0,2,4,6, the relative local rank for these GPUs are still 0, 1, 2, 3

3. Comment out the line of shell command with GPUs that you don't use

4. When use the evaluate.sh script, use 1 GPU is sufficient for current project. More GPU doesn't necessarily faster.

5. This script expand the iterative for loop to different section, so that you can easily resart/continue the algorithm 
    is unexpected case happened, i.e. CUDA out of mem.

6. Evaluation result are saved into eval_result dir in the RAFT dir

7. The evaluation will load data from RAFT/data/test

8. The sanity_check parameter represents a test run with small amount of data if set to 1, default to 0

Remember to comment out the corresponding lines of code of the GPUs that don't use.

For example, if only use 4 GPUs, then only use the shell command of assigned GPUSs

i.e. If use 0,2,4,6, then comment out the code of 1,3,5,7. Same apply to get_rewards.sh

And notice the local rank at the end of each line is still 0, 1, 2, 3

```bash
num_gpus=4
gpu_list=0,2,4,6
CUDA_VISIBLE_DEVICES=0 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 0 ${random_seed} &

#CUDA_VISIBLE_DEVICES=1 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 1 ${random_seed} &

CUDA_VISIBLE_DEVICES=2 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 1 ${random_seed} &

#CUDA_VISIBLE_DEVICES=3 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 3 ${random_seed} &

CUDA_VISIBLE_DEVICES=4 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 2 ${random_seed} &

#CUDA_VISIBLE_DEVICES=5 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 5 ${random_seed} &

CUDA_VISIBLE_DEVICES=6 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 3 ${random_seed} &

#CUDA_VISIBLE_DEVICES=7 bash ./get_samples.sh ${sft_model} 0 ${num_gpus} ${model_dir}/infer_set ${sanity_check} 7 ${random_seed} &
```

### Example of configuration of restart_raft.sh
```bash
# specify the base_dir as where to store the models during iterative process, 
base_dir="iterative_llama_pool_prm"
mkdir -p $base_dir
# You should edit the sft model dir accordingly
sft_model="meta-llama/Meta-Llama-3-8B-Instruct"
reward_model="CarelessLee/MCQ_pooled_full_rationale_confidence_predictor"

# set to 1 if want to verify the pipeline with small amount of data
sanity_check=0
# specify the number and which GPUs to use
num_gpus=4
gpu_list=0,1,2,3

# set random seed
random_seed=42
```


## TODO
* Optimize and add ISP-related code
* Optimize code for MCQ training set generator



