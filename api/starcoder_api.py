import logging
from transformers import logging as hf_logging

logging.basicConfig(level=logging.INFO)
hf_logging.set_verbosity_info()

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS
import re

app = Flask(__name__)

# Allow all subdomains of black-ide.space and your ngrok URL
CORS(app, resources={r"/*": {"origins": [
    re.compile(r"https?://([a-z0-9-]+\.)*black-ide\.space"),
    "https://66df2246a5c7.ngrok-free.app"
]}})

# Load model and tokenizer
MODEL_ID = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.json
    code = data.get('code', '')
    # Prepend instruction to steer the model toward practical Python completions
    prompt = '"""Complete the following Python code without explanations.\n\n' + code
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    gen_tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=80,
        min_new_tokens=12,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        repetition_penalty=1.3
    )
    completion = tokenizer.decode(gen_tokens[0][input_ids.shape[-1]:], skip_special_tokens=True)
    # Remove comments, markdown, and blank lines
    python_lines = [
        line for line in completion.split('\n')
        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('```')
    ]
    python_code = '\n'.join(python_lines).strip() or completion.strip()
    return jsonify({'completion': python_code})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)