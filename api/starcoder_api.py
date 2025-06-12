from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

STARCODER_ENDPOINT = "http://localhost:5001/autocomplete"

@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    data = request.get_json()
    if not data or "code" not in data:
        return jsonify({"error": "Missing 'code' in request"}), 400

    payload = {
        "code": data["code"],
        "max_tokens": data.get("max_tokens", 64),
        "stop_token": data.get("stop_token", None),
    }
    try:
        response = requests.post(STARCODER_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return jsonify({"completion": result.get("completion", "")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)