# /api/tensorflow.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

model_tensorflow = None
tokenizer = None
max_len = 1128

def load_model_and_tokenizer():
    global model_tensorflow, tokenizer
    if model_tensorflow is None or tokenizer is None:
        try:
            with open("tokenizer.pkl", "rb") as handle:
                tokenizer = pickle.load(handle)
            model_tensorflow = load_model("best_model.keras")
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            raise RuntimeError("Failed to load TensorFlow model")

@app.route("/analyze", methods=["POST"])
def analyze_sentiment_tensorflow():
    try:
        load_model_and_tokenizer()
    except RuntimeError:
        return jsonify({"error": "Model failed to load"}), 500

    data = request.get_json()
    text = data.get("review", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="pre")
    prediction = model_tensorflow.predict(padded_sequence)[0]
    sentiment = "POSITIVE" if prediction[1] > 0.5 else "NEGATIVE"
    confidence = float(prediction[1] if sentiment == "POSITIVE" else prediction[0])

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence,
        "message": f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})"
    })
