# /api/tfidf.py
from flask import Flask, request, jsonify
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model_lr = None
vectorizer = None

def load_model():
    global model_lr, vectorizer
    if model_lr is None or vectorizer is None:
        try:
            model_lr = joblib.load("sentiment_model.pkl")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
        except Exception as e:
            print(f"Error loading TF-IDF model: {e}")
            raise RuntimeError("Failed to load TF-IDF model")

# Ensure NLTK data is downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)

@app.route("/analyze", methods=["POST"])
def analyze_sentiment_tfidf():
    try:
        load_model()
    except RuntimeError:
        return jsonify({"error": "Model failed to load"}), 500

    data = request.get_json()
    text = data.get("review", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed_review = preprocess_text(text)
    review_tfidf = vectorizer.transform([processed_review])
    prediction = model_lr.predict(review_tfidf)[0]
    probability = model_lr.predict_proba(review_tfidf)[0]
    confidence = max(probability)

    return jsonify({
        "sentiment": prediction,
        "confidence": float(confidence),
        "message": f"Predicted Sentiment: {prediction} (Confidence: {confidence:.4f})"
    })
