import torch
import torch.nn as nn

class GPTConfig:
    def __init__(self, vocab_size, n_layer=6, n_head=8, n_embd=512, block_size=512):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        token_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(token_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
