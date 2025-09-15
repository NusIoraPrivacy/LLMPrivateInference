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
from utils.other_utils import get_unique_fake_attrs, get_sample_number

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

models = ["FacebookAI/roberta-base", "Qwen/Qwen2.5-0.5B-Instruct"]

def get_model_tokenizer(model_name, args):
    if "Qwen" in model_name or "Llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map = "auto")
    # elif "bart" in model_name or "t5" in model_name:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForSequenceClassification.from_pretrained(model_name).to(args.device)
    # elif "gpt2" in model_name:
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
    parser.add_argument("--model_name", type=str, default=models[1])
    parser.add_argument("--test_only", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thd", type=float, default=0.5)
    parser.add_argument("--decay_weight", type=float, default=0.1)
    parser.add_argument("--sample_mul", type=float, default=1)
    parser.add_argument("--max_length", type=int, default=200) # cpr: 100; full: 250
    args = parser.parse_args()
    return args

def get_probs(model, tokenizer, org_attrs, fake_attr_list, org_query, loss_function, args):
    snippet_list = []
    snippet = org_query
    snippet_list = []
    for fake_attrs in fake_attr_list:
        this_snippet = "" + snippet
        # print(this_snippet)
        for org_attr, fake_attr in zip(org_attrs, fake_attrs):
            this_snippet = this_snippet.replace(org_attr, fake_attr)
        snippet_list.append(this_snippet)
    inputs = tokenizer(snippet_list, return_tensors="pt", padding="longest")
    # print(inputs['input_ids'][0])
    # print(inputs['input_ids'].shape)
    if inputs['input_ids'].shape[-1] > args.max_length:
        inputs = tokenizer(snippet_list, return_tensors="pt", padding="max_length", max_length=args.max_length, truncation=True)
    # print(inputs['input_ids'].shape)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    # input_ids = inputs['input_ids'].to(model.device)
    # target_ids[:, :start_idx] = -100
    # print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        probs = (probs[...,0]).tolist()

    return probs

def filter_attributes(sample, model, tokenizer, loss_function, args):
    priv_attrs, fake_attrs, query = sample["private attributes"], sample["fake attributes"], sample["question"]
    if len(fake_attrs)==0 or len(query) == 0:
        return None, None

    # sample fake attributes
    n_samples = get_sample_number(fake_attrs, ratio=1/args.sample_mul)
    beam = []
    sample_fake_attrs = []
    for org_attr, fake_attr_list in zip(priv_attrs, fake_attrs):
        fake_attr_list = list(set(fake_attr_list))
        sample_fakes = random.choices(fake_attr_list, k = n_samples)
        sample_fake_attrs.append(sample_fakes)
    sample_fake_attrs = get_unique_fake_attrs(sample_fake_attrs)
    # print("attribute length:", len(priv_attrs))
    
    itrs = len(sample_fake_attrs)//args.batch_size + 1
    scores = []
    for i in range(itrs):
        this_seq_list = sample_fake_attrs[(i*args.batch_size):((i+1)*args.batch_size)]
        if len(this_seq_list) > 0:
            this_scores = get_probs(model, tokenizer, priv_attrs, this_seq_list, query, loss_function, args)
            scores.extend(this_scores)
    
    sample_fake_combs = []
    min_score = 100
    best_comb = None
    for score, seq in zip(scores, sample_fake_attrs):
        if score < args.thd:
            sample_fake_combs.append(seq)
        if score < min_score:
            best_comb = seq
    if len(sample_fake_combs) == 0:
        sample_fake_combs.append(best_comb)
    return sample_fake_attrs, sample_fake_combs

if __name__ == "__main__":
    args = parse_args()
    print(args)
    log_root = f"{args.root_path}/result/logs"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    model_name = (args.model_name).split("/")[-1]
    file_name = f"gen-{args.sample_mul}-{model_name}.log"
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

    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    out_dir = f'{args.root_path}/result/{args.data_name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = f'{out_dir}/fake_attr_{model_name}_{args.sample_mul}.json'

    outputs = []

    total_time = 0
    total_cnt = 0
    all_samples = []
    total_sample_sizes = []
    t1 = time.time()
    with tqdm(total=len(data)) as pbar:
        for cnt, sample in enumerate(data):

            sample_fake_attrs, sample_fake_combs = filter_attributes(sample, model, tokenizer, loss_function, args)
            if sample_fake_combs is None:
                pbar.update(1)
                continue

            # sample fake attributes
            sample["sample fake combinations"] = sample_fake_combs
            n_samples = len(sample_fake_combs)
            n_total_samples = len(sample_fake_attrs)
            all_samples.append(n_samples)
            total_sample_sizes.append(n_total_samples)
            outputs.append(sample)
            
            t2 = time.time()
            total_time = t2 - t1
            total_cnt += 1
            pbar.update(1)
            avg_time = total_time/total_cnt
            avg_sample_size = sum(all_samples)/len(all_samples)
            avg_total_sample = sum(total_sample_sizes)/len(total_sample_sizes)
            pbar.set_postfix(time=avg_time, n_sample=n_samples, total_n_sample=n_total_samples)
            logging.info(
                f"Iteration {cnt+1}/{len(data)} - "
                f"Average total sample size {avg_total_sample} - "
                f"Average sample size {avg_sample_size} - "
                f"Average time {avg_time}s"
            )

    with open(out_path, 'w') as fout:
        json.dump(outputs, fout, indent=4)