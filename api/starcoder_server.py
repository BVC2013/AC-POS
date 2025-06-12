from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Choose a model you have resources for.
# For demo/testing: 'bigcode/starcoderbase-1b' (small), for full: 'bigcode/starcoder'
model_id = "bigcode/starcoderbase-3b"  # Or 'bigcode/starcoder'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.get_json()
    code = data.get('code', '')
    max_tokens = int(data.get('max_tokens', 64))
    stop_token = data.get('stop_token', None)

    if not code:
        return jsonify({"error": "Missing code"}), 400

    # Use tokenizer to get both input_ids and attention_mask
    inputs = tokenizer(code, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  # <-- Add this line
            max_length=input_ids.shape[1] + max_tokens,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    if stop_token and stop_token in generated:
        generated = generated.split(stop_token)[0]

    return jsonify({"completion": generated})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)