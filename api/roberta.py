# /api/roberta.py
from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

model_roberta = None
tokenizer_roberta = None

def load_model_and_tokenizer():
    global tokenizer_roberta, model_roberta
    if tokenizer_roberta is None or model_roberta is None:
        try:
            tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            model_roberta = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            print(f"Error loading RoBERTa model: {e}")
            raise RuntimeError("Failed to load RoBERTa model")

@app.route("/analyze", methods=["POST"])
def analyze_sentiment_roberta():
    try:
        load_model_and_tokenizer()
    except RuntimeError:
        return jsonify({"error": "Model failed to load"}), 500

    data = request.get_json()
    text = data.get("review", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        inputs = tokenizer_roberta(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model_roberta(**inputs)
            scores = outputs.logits.softmax(dim=1)
            prediction = scores.argmax().item()

        sentiment_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        sentiment = sentiment_map[prediction]
        confidence = float(scores[0][prediction].item())

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence,
            "message": f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
