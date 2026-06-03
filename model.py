import torch
import torch.nn as nn

# -----------------------------
# Transformer Block
# -----------------------------
class Block(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


# -----------------------------
# Tiny GPT (~1M params target)
# -----------------------------
class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size=5000,
        d_model=128,
        n_layers=4,
        max_len=64
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.Sequential(
            *[Block(d_model) for _ in range(n_layers)]
        )

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(0, T, device=x.device).unsqueeze(0)

        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)


model = TinyGPT()

params = sum(p.numel() for p in model.parameters())
print("Total parameters:", params)
