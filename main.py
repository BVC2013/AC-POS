# GPT Code Completion Model for Local Training on 12GB GPU

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch.utils.checkpoint as checkpoint

# ----- GPT Configuration -----

class GPTConfig:
    def __init__(self, vocab_size, n_layer=12, n_head=12, n_embd=768, block_size=1024):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

# ----- GPT Model -----

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        token_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(token_emb + pos_emb)

        for block in self.blocks:
            x = checkpoint.checkpoint(block, x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ----- Tokenizer & Dataset -----

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
block_size = 1024

def tokenize(example):
    return tokenizer(example['content'], truncation=True, max_length=block_size)

dataset = load_dataset("codeparrot/codeparrot-clean", split="train")
dataset = dataset.shuffle(seed=42).select(range(50000))  # Expandable as needed
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=['content'])

# ----- Collate Function -----

def collate_fn(batch):
    input_ids = [torch.tensor(x['input_ids']) for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets = input_ids.clone()
    return input_ids, targets

train_loader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# ----- Training Loop -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = GPTConfig(vocab_size=tokenizer.vocab_size)
model = GPT(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
scaler = torch.cuda.amp.GradScaler()

model.train()
for epoch in range(3):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} complete. Avg Loss: {total_loss/len(train_loader):.4f}")

# ----- Save Full Model -----
torch.save(model.state_dict(), "gpt-codeparrot.pt")

# ----- Optional: Quantize for Deployment -----
quantized_model = torch.quantization.quantize_dynamic(
    model.cpu(), {nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), "gpt-codeparrot-int8.pt")