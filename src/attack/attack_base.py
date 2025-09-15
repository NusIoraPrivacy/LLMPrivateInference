from transformers import default_data_collator, get_scheduler
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from sklearn.metrics import f1_score

from tqdm import tqdm
import random
import json
import os
import argparse
import numpy as np
import logging

from utils.data_utils import AttackDatasetTrain, AttackDatasetTest
from utils.model_utils import get_model_tokenizer
from utils.other_utils import get_unique_fake_attrs

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

attack_models = ["meta-llama/Llama-3.2-1B", "FacebookAI/roberta-large"]
discrim_models = ["FacebookAI/roberta-base", "Qwen/Qwen2.5-0.5B-Instruct"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--token_len", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--discrim_model", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--attack_model", type=str, default=attack_models[0])
    parser.add_argument("--sample_mul", type=float, default=1)
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_word", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use_peft", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    log_root = f"{args.root_path}/result/logs"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    attack_model_name = (args.attack_model).split("/")[-1]
    discrim_model_name = (args.discrim_model).split("/")[-1]
    file_name = f"id-attack-base-{attack_model_name}-{args.data_name}.log"
    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    model, tokenizer = get_model_tokenizer(args.attack_model, num_labels=2, args=args)
    # print(model)
    discrim_model_name = "-".join((args.discrim_model).split("/"))
    data_path = f'{args.root_path}/result/{args.data_name}/fake_attr_{discrim_model_name}_{args.sample_mul}.json'
    with open(data_path) as fin:
        raw_data = json.load(fin)

    # create classification dataset 
    process_data = []
    for sample in raw_data:
        priv_attrs, fake_attrs, fake_attr_combs, query = sample["private attributes"], sample["fake attributes"], sample["sample fake combinations"], sample["question"]
        # randomly sample one combination as the fake query
        if len(fake_attrs) > 0:
            this_queries = []
            this_queries.append({"prompt": query, "label": 1})
            sample_fake_attrs = []
            for fake_attr_list in fake_attrs:
                fake_attr_list = list(set(fake_attr_list))
                sample_fakes = random.choices(fake_attr_list, k=len(fake_attr_combs))
                sample_fake_attrs.append(sample_fakes)
            sample_fake_attrs = get_unique_fake_attrs(sample_fake_attrs)
            # print(sample_fake_attrs)
            for fake_attr_list in sample_fake_attrs:
                this_query = "" + query
                for priv_attr, fake_attr in zip(priv_attrs, fake_attr_list):
                    this_query = this_query.replace(priv_attr, fake_attr)
                this_queries.append({"prompt": this_query, "label": 0})
            process_data.append(this_queries)

    random.shuffle(process_data)
    n_train = int(len(process_data) * args.train_pct)
    train_data = process_data[:n_train]
    test_data = process_data[n_train:]
    train_dataset = AttackDatasetTrain(train_data, tokenizer, args.max_word)
    test_dataset = AttackDatasetTest(test_data, tokenizer, args.max_word)
    # obtain dataloader
    train_loader = DataLoader(
            train_dataset, 
            batch_size=args.train_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )

    # prepare optimizer and scheduler
    optimizer = optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=0.0,
                )
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
        )
    
    # train models
    for epoch in range(args.epochs):
        model.train()
        loss_list = []

        with tqdm(total=len(train_loader)) as pbar:
            for step, batch in enumerate(train_loader):
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                # print(batch)
                output = model(input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"],
                            labels = batch["labels"]) 
                loss = output.loss
                loss_list.append(loss.item())
                loss_avg = sum(loss_list)/len(loss_list)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss_avg)
                # break
            print(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')
        
        model.eval()
        asr_list = []
        f1_list = []
        with tqdm(total=len(test_loader)) as pbar:
            for i, batch in enumerate(test_loader):
                for key in batch:
                    batch[key] = batch[key].to(model.device)
                with torch.no_grad():
                    outputs = model(
                            input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"])
                logits = outputs.logits
                y_pred = torch.argmax(logits, -1).tolist()
                success = 0
                if y_pred[0] == 1:
                    success = 1
                    for y in y_pred[1:]:
                        if y == 1:
                            success = 0
                f1 = f1_score(batch["labels"].tolist(), y_pred)
                asr_list.append(success)
                f1_list.append(f1)
                asr = sum(asr_list)/len(asr_list)
                f1 = sum(f1_list)/len(f1_list)
                pbar.update(1)
                pbar.set_postfix(asr=asr, f1=f1)
        asr = sum(asr_list)/len(asr_list)
        f1 = sum(f1_list)/len(f1_list)
        print(f"Attack success rate for epoch {epoch}: {asr}")
        print(f"F1 for epoch {epoch}: {f1}")

        logging.info(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"ASR {asr} - "
                f"F1 {f1}"
            )