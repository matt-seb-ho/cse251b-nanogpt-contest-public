import os
import numpy as np
import tiktoken
from datasets import load_dataset, interleave_datasets
from tqdm import tqdm
from huggingface_hub import login

# use huggingface token if it exists
if os.path.isfile("hf_token.txt"):
    with open("hf_token.txt",'r') as f:
        try:
            login(token=f.read())
            print("HF user token accepted.")
        except:
            print("HF Token invalid. Continuing without...")


# Setup which datasets to grab
local_dir = "common_pile_filtered"
remote_names = [
    ("arxiv_abstracts_filtered",.25),     # paper abstracts
    ("libretexts_filtered",     .50),     # textbooks
    ("doab_filtered",           .25),     # peer-reviewed books
    #"stackexchange_filtered",       # stack exchange Q&A
]
names, probs = zip(*remote_names)
shard_size = int(1e8) #100M tokens
max_shards = 7
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

datasets_list = [
    load_dataset(f"common-pile/{name}", split="train", streaming=True).select_columns(["text"])
    for name in names
]
combined = interleave_datasets(
    datasets_list,
    probabilities=list(probs),
    stopping_strategy="all_exhausted",
    seed=42,
)

# Counting tokens
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Write shards
shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
total_tokens_written = 0
progress_bar = None

for doc in combined:
    if shard_index >= max_shards:
        break

    tokens = tokenize(doc)

    if token_count + len(tokens) < shard_size:
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"commonpile_{split}_{shard_index:06d}")
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder

if token_count != 0 and shard_index < max_shards:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"commonpile_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])