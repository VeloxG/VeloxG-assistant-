from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Model ndogo
MODEL_NAME = "microsoft/DialoGPT-small"

# Load model na tokenizer mara moja tu wakati app inaanza
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Chagua CPU kwa sababu Render free tier haina GPU
device = torch.device("cpu")
model.to(device)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Chatbot is running on Render free tier!"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Tokenize input
        inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt").to(device)

        # Generate response
        outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render inahitaji host='0.0.0.0'
    app.run(host="0.0.0.0", port=5000)
