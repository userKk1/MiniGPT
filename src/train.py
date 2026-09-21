import argparse
import json
import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from tokenizers import Tokenizer

from config import cfg, DATA_PROCESSED_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR
from src.model import GPT


def load_data():
    tokenizer = Tokenizer.from_file(
        str(DATA_PROCESSED_DIR / "tokenizer.json")
    )

    vocab_size = tokenizer.get_vocab_size()

    train_data = np.memmap(
        DATA_PROCESSED_DIR / "train.bin",
        dtype=np.uint16,
        mode="r"
    )

    val_data = np.memmap(
        DATA_PROCESSED_DIR / "val.bin",
        dtype=np.uint16,
        mode="r"
    )

    return train_data, val_data, vocab_size


def get_batch(split, train_data, val_data, device):
    data = train_data if split == "train" else val_data
    block_size = cfg.model.block_size
    batch_size = cfg.train.batch_size
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_lr(step):
    t = cfg.train
    if step < t.warmup_steps:
        return t.learning_rate * (step + 1) / t.warmup_steps
    if step > t.lr_decay_steps:
        return t.min_lr
    decay_ratio = (step - t.warmup_steps) / (t.lr_decay_steps - t.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return t.min_lr + coeff * (t.learning_rate - t.min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.train.eval_iters)
        for k in range(cfg.train.eval_iters):
            x, y = get_batch(split, train_data, val_data, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(fresh: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: training on CPU -- this will be slow. Consider Colab's free GPU.")
    print("device:", device)

    train_data, val_data, vocab_size = load_data()
    print(f"vocab_size: {vocab_size}  train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    model = GPT(vocab_size).to(device)
    print(f"Parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )

    start_step = 0
    history = {"step": [], "train_loss": [], "val_loss": []}
    ckpt_path = CHECKPOINTS_DIR / "ckpt.pt"
    if ckpt_path.exists() and not fresh:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resuming from step {start_step}")

    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb = get_batch("train", train_data, val_data, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | {elapsed:.0f}s"
            )
            history["step"].append(step)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "step": step, "val_loss": losses["val"]},
                    CHECKPOINTS_DIR / "ckpt_best.pt",
                )

        if step % cfg.train.checkpoint_every == 0 and step > 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                ckpt_path,
            )

    print("Training done.")

    with open(LOGS_DIR / "loss_log.json", "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["step"], history["train_loss"], label="train")
    plt.plot(history["step"], history["val_loss"], label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend()
    plt.savefig(FIGURES_DIR / "loss_curve.png", dpi=150, bbox_inches="tight")
    print(f"Saved loss curve to {FIGURES_DIR / 'loss_curve.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint and start over")
    args = parser.parse_args()
    train(fresh=args.fresh)
