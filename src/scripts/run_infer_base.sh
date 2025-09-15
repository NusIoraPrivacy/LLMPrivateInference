#!/bin/bash

gpt_models=("gpt-4.1-mini" "gpt-3.5-turbo")

for gpt_model in ${gpt_models[@]}
do
    echo "Query online model $gpt_model for non-private setting"
    python -m online_infer.base_infer --query_online_model $gpt_model --data_size 100
done