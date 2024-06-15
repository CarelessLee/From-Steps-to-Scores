# From Steps to Scores

This is the repo for paper 'From Steps to Scores: Improving LLM Certainty and Predicting Uncertainty in Math Reasoning'.



## Introduction

We present a novel framework to quantify and enhance the reliability of large language models (LLMs) in solving complex mathematical problems that require multi-step reasoning. Our approach is structured into **three stages**. First, we introduced “Iterative Stepwise Prompting” (**ISP**), a technique that incrementally reveals and refines reasoning steps to LLMs to improve their certainty. Second, we developed the “Mathematical Confidence Quantifier” (**MCQ**), a Reinforcement Learning-based prediction model to assign certainty scores to each mathematical reasoning step, offering a more granular confidence assessment. Finally, we employed Proximal Policy Optimization (PPO) using MCQ to further enhance the reliability and performance of LLMs. Experiments on the MathQA dataset using Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, and the more advanced GPT-4o models demonstrated considerable gains in accuracy and reduced uncertainty for the ISP phase. This combined methodology provides a robust framework for enhancing LLMs’ performance and reliability in mathematical problem-solving tasks, advancing the development of more trustworthy and capable LLMs.

<img width="561" alt="Screenshot 2024-06-03 at 11 17 16 PM" src="https://github.com/CarelessLee/From-Steps-to-Scores/assets/42497570/22413247-7a78-448c-99f9-4275088fe892">


The repo currently consists of Stage 1 and Stage 2's work for the Certainty Predictor (MCQ) models and PPO/DPO RL models. Directory **MQADatasetsConstruction** consists of scripts that generates training sets for MCQ, and **UncertaintyExperiments** consists of work for training & testing MCQ and training DPO.


## Getting Start

``` 
git clone git@github.com:CarelessLee/From-Steps-to-Scores.git

cd From-Steps-to-Scores
```



## Requirements

The **LMFlow** environment contains all the packages and dependencies required.

```  
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

```
# put MathQA json file in the UncertaintyExperiments directory and config 'dataset_path' in mathqa_run_prediction.sh accordingly
cd UncertaintyExperiments
./mathqa_run_prediction.sh
```



This will create a directory 'mathqa_model' that stores the parameters of the trained MCQ model. You can testout MCQ's response using **mathqa_prediction_model_train.py**



## DPO

with the trained MCQ, we can now reinforce LLMs with the help of RLHF algorithms. Here we demonstrate the usecase of DPO. Remember to load the pre-processed datasets (e.g. fully_processed_deepmind_responses.json) into the **UncertaintyExperiments** directory beforehand.

```
./new_run_dpo.sh
```





