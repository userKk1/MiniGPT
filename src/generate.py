import argparse

import torch
from tokenizers import Tokenizer

from config import cfg, CHECKPOINTS_DIR, DATA_PROCESSED_DIR
from src.model import GPT


def load_tokenizer():
    tokenizer_path = DATA_PROCESSED_DIR / "tokenizer.json"
    return Tokenizer.from_file(str(tokenizer_path))


def load_model(checkpoint_name: str, device: str):
    # vocab_size is baked into the checkpoint's embedding table shape, so we
    # can recover it directly instead of also depending on vocab.json here.
    ckpt = torch.load(CHECKPOINTS_DIR / checkpoint_name, map_location=device)
    vocab_size = ckpt["model"]["token_embedding.weight"].shape[0]

    model = GPT(vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss")
    if val_loss is not None:
        print(f"Loaded {checkpoint_name} (step {step}, val_loss={val_loss:.4f})")
    else:
        print(f"Loaded {checkpoint_name} (step {step})")

    return model


def main():
    parser = argparse.ArgumentParser(description="Generate code completions from MiniGPT.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to complete.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="ckpt_best.pt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.top_k is not None and args.top_p is not None:
        raise ValueError("--top_k and --top_p are mutually exclusive -- pass only one.")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer()

    idx = torch.tensor([tokenizer.encode(args.prompt).ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()