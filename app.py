from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

MODEL_NAME = "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cpu")
model.to(device)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Smallest AI Chatbot Running"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "Message required"}), 400

        inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=80, pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
