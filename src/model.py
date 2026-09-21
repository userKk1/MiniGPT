import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v   # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feedforward: expand 4x, GELU, project back down."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: LN -> attention -> residual, LN -> FFN -> residual."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        m = cfg.model
        self.block_size = m.block_size

        self.token_embedding = nn.Embedding(vocab_size, m.n_embd)
        self.position_embedding = nn.Embedding(m.block_size, m.n_embd)
        self.blocks = nn.Sequential(
            *[Block(m.n_embd, m.n_head, m.block_size, m.dropout) for _ in range(m.n_layer)]
        )
        self.ln_f = nn.LayerNorm(m.n_embd)
        self.lm_head = nn.Linear(m.n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Standard small-init scheme (GPT-2 style) -- keeps early training stable.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )

        tok_emb = self.token_embedding(idx)                                   # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend idx by max_new_tokens tokens. Used by
        generate.py and for periodic training-time sanity checks."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick smoke test: python src/model.py
    import json

    from config import DATA_PROCESSED_DIR

    vocab_path = DATA_PROCESSED_DIR / "vocab.json"
    if vocab_path.exists():
        vocab_size = json.loads(vocab_path.read_text())["vocab_size"]
    else:
        vocab_size = 100
        print("vocab.json not found -- using dummy vocab_size=100 for smoke test")

    model = GPT(vocab_size)
    print(f"Parameters: {model.num_params():,}")

    dummy_idx = torch.randint(0, vocab_size, (2, cfg.model.block_size))
    dummy_targets = torch.randint(0, vocab_size, (2, cfg.model.block_size))
    logits, loss = model(dummy_idx, dummy_targets)
    print(f"logits shape: {logits.shape}, loss: {loss.item():.4f}")
