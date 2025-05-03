from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Check for NLTK stopwords and download if not available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# ---------------------- Initialize models at startup ----------------------
# Initialize variables
tokenizer = None
model_tensorflow = None
tokenizer_roberta = None
model_roberta = None
model_lr = None
vectorizer = None

# Load TF-IDF Logistic Regression Model
try:
    model_lr = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("TF-IDF Logistic Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading TF-IDF model: {e}")

# Load TensorFlow Model
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model_tensorflow = load_model("best_model.keras")
    max_len = 1128
    print("TensorFlow model loaded successfully.")
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")

# Load HuggingFace RoBERTa Model
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    
    tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model_roberta = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("RoBERTa model loaded successfully.")
except Exception as e:
    print(f"Error loading HuggingFace RoBERTa model: {e}")

# ---------------------- Text preprocessing function ----------------------
def preprocess_text(text):
    """Preprocess text: lowercase, remove special characters, and stopwords."""
    text = text.lower()
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# ---------------------- Sentiment prediction functions ----------------------
def predict_sentiment_tfidf(text):
    """Predict sentiment using the Logistic Regression model."""
    if not model_lr or not vectorizer:
        return "Error: TF-IDF model not loaded properly."
    
    processed_review = preprocess_text(text)
    review_tfidf = vectorizer.transform([processed_review])
    prediction = model_lr.predict(review_tfidf)[0]
    probability = model_lr.predict_proba(review_tfidf)[0]
    confidence = max(probability)
    
    return {
        "sentiment": prediction,
        "confidence": float(confidence),
        "message": f"Predicted Sentiment: {prediction} (Confidence: {confidence:.4f})"
    }

def predict_sentiment_tensorflow(text):
    """Predict sentiment using the TensorFlow model."""
    if not tokenizer or not model_tensorflow:
        return {
            "sentiment": "ERROR",
            "confidence": 0.0,
            "message": "Error: TensorFlow model not loaded properly."
        }
    
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="pre")
        prediction = model_tensorflow.predict(padded_sequence)[0]
        sentiment = "POSITIVE" if prediction[1] > 0.5 else "NEGATIVE"
        confidence = float(prediction[1] if sentiment == "POSITIVE" else prediction[0])
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "message": f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})"
        }
    except Exception as e:
        return {
            "sentiment": "ERROR",
            "confidence": 0.0,
            "message": f"Error in prediction: {e}"
        }

def predict_sentiment_roberta(text):
    """Predict sentiment using the HuggingFace RoBERTa model."""
    if not tokenizer_roberta or not model_roberta:
        return {
            "sentiment": "ERROR",
            "confidence": 0.0,
            "message": "Error: RoBERTa model not loaded properly."
        }
    
    try:
        # Encode the text and prepare for the model
        inputs = tokenizer_roberta(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model_roberta(**inputs)
            scores = outputs.logits.softmax(dim=1)
            prediction = scores.argmax().item()
        
        # Map the prediction to sentiment label
        # The model returns 0 for negative, 1 for neutral, and 2 for positive
        sentiment_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        sentiment = sentiment_map[prediction]
        
        # Get confidence scores
        confidence = float(scores[0][prediction].item())
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "message": f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})"
        }
    except Exception as e:
        return {
            "sentiment": "ERROR",
            "confidence": 0.0,
            "message": f"Error in RoBERTa prediction: {e}"
        }

# ---------------------- Flask routes ----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    if request.method == "POST":
        data = request.get_json()
        text = data.get("review", "")
        model_choice = data.get("model", "RoBERTa Model")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if model_choice == "TF-IDF Logistic Regression":
            result = predict_sentiment_tfidf(text)
        elif model_choice == "TensorFlow Model":
            result = predict_sentiment_tensorflow(text)
        else:  # RoBERTa Model
            result = predict_sentiment_roberta(text)
            
        return jsonify(result)

@app.route("/health")
def health_check():
    health = {
        "status": "ok",
        "models": {
            "tfidf": model_lr is not None and vectorizer is not None,
            "tensorflow": model_tensorflow is not None and tokenizer is not None,
            "roberta": model_roberta is not None and tokenizer_roberta is not None
        }
    }
    return jsonify(health)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)