# config.py

"""
Usage:
    from config import cfg
    print(cfg.model.n_layer)
"""
 
from dataclasses import dataclass, field
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Paths (relative to project root, wherever the repo is cloned)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent  # MiniGPT/
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
 
for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
 
 
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name: str = "codeparrot/codeparrot-clean"

    max_corpus_mb: int = 50
    max_file_chars: int = 20_000

    train_split: float = 0.9
    seed: int = 1337
 
 
# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
@dataclass
class TokenizerConfig:
    type: str = "char"          # "char" for v1, "bpe" for a later upgrade
    vocab_size: int = None       # set automatically once the char vocab is built
 
 
# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 256        # context length (tokens per training sequence)
    dropout: float = 0.1
    bias: bool = True
 
 
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 10000
    eval_interval: int = 250
    eval_iters: int = 100
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    lr_decay_steps: int = 10000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"         # train.py falls back to "cpu" if unavailable
    checkpoint_every: int = 500
 
 
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
 
 
cfg = Config()
 
if __name__ == "__main__":
    # Quick sanity check: python config.py
    print(cfg)
