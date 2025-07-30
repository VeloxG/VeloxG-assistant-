import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "Server running ✅",
        "endpoint": "/chat",
        "model": model_name
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    with torch.no_grad():
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
        reply_ids = model.generate(inputs, max_length=60, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"reply": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
