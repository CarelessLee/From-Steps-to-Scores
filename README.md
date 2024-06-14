# From Steps to Scores

This is the repo for paper 'From Steps to Scores: Improving LLM Certainty and Predicting Uncertainty in Math Reasoning'.


## Introduction

We present a novel framework to quantify and enhance the reliability of large language models (LLMs) in solving complex mathematical problems that require multi-step reasoning. Our approach is structured into **three stages**. First, we introduced “Iterative Stepwise Prompting” (**ISP**), a technique that incrementally reveals and refines reasoning steps to LLMs to improve their certainty. Second, we developed the “Mathematical Confidence Quantifier” (**MCQ**), a Reinforcement Learning-based prediction model to assign certainty scores to each mathematical reasoning step, offering a more granular confidence assessment. Finally, we employed Proximal Policy Optimization (PPO) using MCQ to further enhance the reliability and performance of LLMs. Experiments on the MathQA dataset using Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, and the more advanced GPT-4o models demonstrated considerable gains in accuracy and reduced uncertainty for the ISP phase. This combined methodology provides a robust framework for enhancing LLMs’ performance and reliability in mathematical problem-solving tasks, advancing the development of more trustworthy and capable LLMs.


## Getting Start

```git clone git@github.com:CarelessLee/From-Steps-to-Scores.git```

```cd From-Steps-to-Scores```



