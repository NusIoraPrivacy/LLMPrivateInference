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
from online_infer.filter import filter_attributes
from utils.query_utils import get_response, create_client
from utils.other_utils import num_tokens_from_string

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

models = ["FacebookAI/roberta-base", "Qwen/Qwen2.5-0.5B-Instruct"]

def get_model_tokenizer(model_name, args):
    if "Qwen" in model_name or "Llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map = "auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(args.device)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--in_file_name", type=str, default="fina_fake_qcattr_none_zero.json")
    parser.add_argument("--model_name", type=str, default=models[0])
    parser.add_argument("--test_only", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--thd", type=float, default=0.5)
    parser.add_argument("--decay_weight", type=float, default=0.1)
    parser.add_argument("--sample_mul", type=float, default=1)
    parser.add_argument("--max_length", type=int, default=200) # cpr: 100; full: 250
    parser.add_argument("--query_online_model", type=str, default="gpt-4.1-mini") # cpr: 100; full: 250
    parser.add_argument("--data_size", type=int, default=100)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    log_root = f"{args.root_path}/result/logs"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    model_name = (args.model_name).split("/")[-1]
    file_name = f"confuse-query-{args.query_online_model}-{args.data_name}-{args.sample_mul}-{model_name}.log"
    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    model_name = "-".join((args.model_name).split("/"))
    model_path = f"{args.root_path}/model/{args.data_name}/{model_name}"
    model, tokenizer = get_model_tokenizer(model_path, args)
    with open(f'{args.root_path}/data/{args.data_name}.json') as fin:
        data = json.load(fin)
    random.shuffle(data)
    data = data[:args.data_size]

    query_client = create_client(args.query_online_model)
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    out_dir = f'{args.root_path}/result/{args.data_name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = f'{out_dir}/confuse_query_{args.query_online_model}_{model_name}_{args.sample_mul}.json'

    outputs = []

    total_time = 0
    total_cnt = 0
    t1 = time.time()
    with tqdm(total=len(data)) as pbar:
        for cnt, sample in enumerate(data):
            query = sample["question"]
            sample_fake_attrs, sample_fake_combs = filter_attributes(sample, model, tokenizer, loss_function, args)
            if sample_fake_combs is None:
                pbar.update(1)
                continue

            # sample fake attributes
            sample["sample fake combinations"] = sample_fake_combs
            # n_total_samples = len(sample_fake_attrs)
            # all_samples.append(n_samples)
            # total_sample_sizes.append(n_total_samples)

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