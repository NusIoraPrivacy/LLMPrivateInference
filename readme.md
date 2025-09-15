## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Offline Training
Train a local discrimination model before online inference

```bash
./scripts/run_train_discrim.sh
```

Can edit the --device or --model_name based on the configuration of gpu or model path.

After training, a discrimination model will be stored in the ./model path.

### 3. Online Inference

#### 3.1 Evaluate privacy protection (ASR & F1)

**Step 1: Generate fake queries**

First we generate a file containing the sample fake queries:
```bash
python -m online_infer.filter --model_name FacebookAI/roberta-base --device cuda:0
```

A file named "fake_attr_FacebookAI-roberta-base_1.json" will be created in the result/mmlu_fina folder.

**Step 2: Evaluate attack accuracies for confusion strategy**

We evaluate the attack success rate and F1 for our strategy:
```bash
./scripts/run_attack_confuse.sh
```

The script conducts inference attack using meta-llama/Llama-3.2-1B and FacebookAI/roberta-large as the attack models. The results will be stored in "result/logs/id-attack-attackmodel-{attack model}-discrimmodel-roberta-base-mmlu_fina-1.log" files.

**Step 3: Evaluate attack accuracies for baseline method**

For baseline method, we do not utilize the discrimination model to choose the fake queries. Instead, we randomly sample a set of queries from the attribute database.

The attack success rate and F1 is obtained from:
```bash
./scripts/run_attack_base.sh
```
The script conducts inference attack using meta-llama/Llama-3.2-1B and FacebookAI/roberta-large as the attack models. The results will be stored in "result/logs/id-attack-base-{attack model}-mmlu_fina.log" files.

#### 3.2 Evaluate time and throughput
**Step 1: Query time & troughput for our method:**
```bash
./scripts/run_infer_confuse.sh
```

The script will output query time and throughput for gpt-4.1-mini and gpt-3.5-turbo models, with time results stored in the "./result/logs/confuse-query-{query_model}-mmlu_fina-1-roberta-base.log" file.

For adaptation to changan setting, can change the --query_online_model to "changan", and modify the functions (create_client & get_response) in ./utilts/query_utils.py.

**Step 2: Query time & troughput for non-private setting:**
```bash
./scripts/run_infer_base.sh
```
The script will output query time and throughput for gpt-4.1-mini and gpt-3.5-turbo models, with time results stored in the "./result/logs/base-query-mmlu_fina-{query_model}.log" file.

For adaptation to changan setting, can change the --query_online_model to "changan".