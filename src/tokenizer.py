import sys
import json

sys.path.append(".")

import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from config import cfg, DATA_RAW_DIR, DATA_PROCESSED_DIR

corpus_path = DATA_RAW_DIR / 'corpus_raw.txt'
text = corpus_path.read_text(encoding='utf-8')
print(f'Corpus length: {len(text):,} characters')

VOCAB_SIZE=1000

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)

tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<UNK>"],
)

print("Training BPE tokenizer...")

tokenizer.train(
    [str(corpus_path)],
    trainer
)

vocab_size = tokenizer.get_vocab_size()

print(f"BPE vocab size: {vocab_size:,}")

print("\nEncoding entire corpus...")

encoded = tokenizer.encode(text)

ids = encoded.ids

print(f"Total BPE tokens: {len(ids):,}")

data_arr = np.array(ids, dtype=np.uint16)

# uint16 supports token IDs from 0 to 65,535.
# Our vocabulary (1000) is far below that limit.

n = len(data_arr)

split_idx = int(n * cfg.data.train_split)

train_ids = data_arr[:split_idx]
val_ids = data_arr[split_idx:]

print(f"train: {len(train_ids):,} tokens")
print(f"val:   {len(val_ids):,} tokens")

train_path = DATA_PROCESSED_DIR / "train.bin"
val_path = DATA_PROCESSED_DIR / "val.bin"

train_ids.tofile(train_path)
val_ids.tofile(val_path)

tokenizer_path = DATA_PROCESSED_DIR / "tokenizer.json"

tokenizer.save(str(tokenizer_path))


metadata = {
    "vocab_size": vocab_size,
    "tokenizer_type": "BPE",
    "pre_tokenizer": "ByteLevel",
    "add_prefix_space": False,
}

with open(
    DATA_PROCESSED_DIR / "tokenizer_config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nDone.")
print(f"Saved train data: {train_path}")
print(f"Saved val data:   {val_path}")
print(f"Saved tokenizer:  {tokenizer_path}")