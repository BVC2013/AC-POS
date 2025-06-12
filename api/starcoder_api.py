from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model and tokenizer
model_id = "bigcode/starcoderbase-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    data = request.get_json()
    code = data.get("code", "")
    max_tokens = data.get("max_tokens", 20)

    # Tokenize with padding and get attention_mask
    inputs = tokenizer(code, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)         # <-- move to device
    attention_mask = inputs["attention_mask"].to(device)  # <-- move to device

    # Generate completion
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    # Get only the generated part
    completion = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return jsonify({"completion": completion})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)