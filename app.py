from flask import Flask, request, jsonify
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

# Tumia GPU kama ipo kwenye Render (CPU tu)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "facebook/blenderbot-90M"  # ndogo kwa hosting
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to(device)

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "Server running", "endpoint": "/chat"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    inputs = tokenizer([user_input], return_tensors="pt").to(device)
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
