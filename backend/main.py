from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer
from model import GPT, GPTConfig

app = FastAPI()

# Allow CORS for frontend localhost access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this!
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
config = GPTConfig(vocab_size=tokenizer.vocab_size)
model = GPT(config)
model.load_state_dict(torch.load("gpt-codeparrot.pt", map_location="cpu"))
model.eval()

class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 20

@app.post("/complete")
def complete(req: CompletionRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    completion = generated[len(req.prompt):]
    return {"completion": completion}
