#!/bin/bash

attack_models=("meta-llama/Llama-3.2-1B" "FacebookAI/roberta-large")

# export CUDA_VISIBLE_DEVICES=0

for attack_model in ${attack_models[@]}
do
    echo "Evaluate attack model $attack_model for base attack"
    python -m attack.attack_base --attack_model $attack_model
done

# echo "Evaluate attack model meta-llama/Llama-3.2-1B discrimation model Qwen/Qwen2.5-0.5B-Instruct"
# python -m attack.attack_confuse --discrim_model "Qwen/Qwen2.5-0.5B-Instruct" --attack_model "meta-llama/Llama-3.2-1B"