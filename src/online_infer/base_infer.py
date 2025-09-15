from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
from accelerate import infer_auto_device_map, init_empty_weights
import torch.nn.functional as F

import logging
import time
from tqdm import tqdm
import random
import json
import os
import argparse
import re
from utils.query_utils import get_response, create_client
from utils.other_utils import num_tokens_from_string

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

models = ["FacebookAI/roberta-base", "Qwen/Qwen2.5-0.5B-Instruct"]
online_models = ["gpt-4.1-mini", "gpt-3.5-turbo"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--query_online_model", type=str, default=online_models[1]) # cpr: 100; full: 250
    parser.add_argument("--data_size", type=int, default=500)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    log_root = f"{args.root_path}/result/logs"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    file_name = f"base-query-{args.data_name}-{args.query_online_model}.log"
    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    with open(f'{args.root_path}/data/{args.data_name}.json') as fin:
        data = json.load(fin)
    random.shuffle(data)
    data = data[:args.data_size]
    query_client = create_client(args.query_online_model)
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    out_dir = f'{args.root_path}/result/{args.data_name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = f'{out_dir}/base_query_{args.query_online_model}_{args.data_name}.json'
    outputs = []

    total_time = 0
    total_cnt = 0
    t1 = time.time()
    with tqdm(total=len(data)) as pbar:
        for cnt, sample in enumerate(data):
            query = sample["question"]

            # query online llm
            prompt = f"Please answer the following question with detailed explanation: {query} "
            answer = get_response(query_client, prompt, args.query_online_model)
            sample["llm response"] = answer
            outputs.append(sample)
            
            t2 = time.time()
            total_time = t2 - t1
            total_cnt += 1
            pbar.update(1)
            avg_time = total_time/total_cnt
            pbar.set_postfix(time=avg_time)
            logging.info(
                f"Iteration {cnt+1}/{len(data)} - "
                f"Average time {avg_time}s"
            )
    # compute throughput
    total_tokens = 0
    for sample in outputs:
        answer = sample["llm response"]
        n_tokens = num_tokens_from_string(answer)
        total_tokens += n_tokens
    throughput = total_tokens/total_time
    print(f"Average throughput: {throughput} tokens/second")

    with open(out_path, 'w') as fout:
        json.dump(outputs, fout, indent=4)