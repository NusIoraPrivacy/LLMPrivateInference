import math
import tiktoken

def get_sample_number(fake_attrs, conf=0.95, ratio=0.5):
    n_attrs = len(fake_attrs)
    t = max(int(n_attrs * 0.2), 1)
    n_candidates = 0
    for fake_attr_list in fake_attrs:
        n_candidates = max(n_candidates, len(fake_attr_list))
    n_comb = math.comb(n_attrs, t)
    n_samples = int((math.log(1-conf)-math.log(n_comb))/math.log(1-1/n_candidates)/ratio)
    return n_samples

def get_unique_fake_attrs(fake_attrs):
    seen = set()
    new_fake_attrs = []
    for combination in zip(*fake_attrs):
        if combination not in seen:
            seen.add(combination)
            new_fake_attrs.append(list(combination))
    return new_fake_attrs

def num_tokens_from_string(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens