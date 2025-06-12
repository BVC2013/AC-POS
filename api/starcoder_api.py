from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    r"https://*.black-ide.space",
    r"http://*.black-ide.space",
    "https://black-ide.space",
    "http://black-ide.space",
    "https://f750-74-88-100-197.ngrok-free.app"
]}})

# Load model and tokenizer
model_id = "bigcode/starcoderbase-1b"  # Change to another StarCoder model if you have access/VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # <-- Add this line
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    try:
        data = request.get_json(force=True)
        code = data.get("code", "")
        max_tokens = int(data.get("max_tokens", 20))

        if not code:
            return jsonify({"error": "No code provided"}), 400

        # Tokenize and move tensors to the correct device
        inputs = tokenizer(code, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

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
    except Exception as e:
        print("Error in /autocomplete:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)