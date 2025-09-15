#!/bin/bash

attack_models=("meta-llama/Llama-3.2-1B" "FacebookAI/roberta-large")
discrim_models=("FacebookAI/roberta-base")

for attack_model in ${attack_models[@]}
do
    for discrim_model in ${discrim_models[@]}
    do
        echo "Evaluate attack model $attack_model discrimation model $discrim_model"
        python -m attack.attack_confuse --discrim_model $discrim_model --attack_model $attack_model
    done
done