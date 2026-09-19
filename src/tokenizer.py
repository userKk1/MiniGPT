import sys
sys.path.append('.')

from config import cfg, DATA_RAW_DIR, DATA_PROCESSED_DIR
import json
import numpy as np

corpus_path = DATA_RAW_DIR / 'corpus_raw.txt'
text = corpus_path.read_text(encoding='utf-8')
print(f'Corpus length: {len(text):,} characters')

chars = sorted(set(text))
vocab_size = len(chars)
print(f'Vocab size: {vocab_size}')

stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

def encode(s:str)->list[int]:
    return [stoi[c] for c in s]

def decode(ids:list[int])->str:
    return ''.join(itos[i] for i in ids)

ids = encode(text)
data_arr = np.array(ids, dtype=np.uint16)  # fine as long as vocab_size < 65536

n = len(data_arr)

split_idx = int(n * cfg.data.train_split)
train_ids = data_arr[:split_idx]
val_ids = data_arr[split_idx:]

print(f'train: {len(train_ids):,} tokens')
print(f'val:   {len(val_ids):,} tokens')

train_ids.tofile(DATA_PROCESSED_DIR / 'train.bin')
val_ids.tofile(DATA_PROCESSED_DIR / 'val.bin')

with open(DATA_PROCESSED_DIR / 'vocab.json', 'w', encoding='utf-8') as f:
    json.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f,ensure_ascii=False, indent=2)

print(f'Saved to {DATA_PROCESSED_DIR}')