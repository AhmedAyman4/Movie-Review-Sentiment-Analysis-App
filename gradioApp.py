import gradio as gr
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Download stopwords if not available
nltk.download("stopwords")

# ---------------------- Load HuggingFace RoBERTa Model ----------------------
try:
    tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model_roberta = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("RoBERTa model loaded successfully.")
except Exception as e:
    print(f"Error loading HuggingFace RoBERTa model: {e}")
    tokenizer_roberta = None
    model_roberta = None

# ---------------------- Load TensorFlow Model ----------------------
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model_tnsorflow = load_model("best_model.keras")
    max_len = 1128
except Exception as e:
    print(f"Error loading TensorFlow model or tokenizer: {e}")
    tokenizer = None
    model_tnsorflow = None

# ---------------------- TensorFlow Sentiment Prediction ----------------------
def predict_sentiment_tensorflow(text):
    """Predict sentiment using the TensorFlow model."""
    try:
        if not tokenizer or not model_tnsorflow:
            return "Error: Model or Tokenizer not loaded properly."
        
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="pre")
        prediction = model_tnsorflow.predict(padded_sequence)[0]
        sentiment = "POSITIVE" if prediction[1] > 0.5 else "NEGATIVE"
        return sentiment
    except Exception as e:
        return f"Error in prediction: {e}"

# ---------------------- HuggingFace RoBERTa Sentiment Prediction ----------------------
def predict_sentiment_roberta(text):
    """Predict sentiment using the HuggingFace RoBERTa model."""
    try:
        if not tokenizer_roberta or not model_roberta:
            return "Error: RoBERTa Model or Tokenizer not loaded properly."
        
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
        
        # Get confidence scores for more detailed results
        confidence = scores[0][prediction].item()
        return f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})"
    except Exception as e:
        return f"Error in RoBERTa prediction: {e}"

# ---------------------- Load & Preprocess Dataset ----------------------
# Paths to datasets
train_path = r"train_data.csv"

# Load training data
train_df = pd.read_csv(train_path)

def preprocess_text(text):
    """Preprocess text: lowercase, remove special characters, and stopwords."""
    text = text.lower()
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Apply text preprocessing
train_df["cleaned_review"] = train_df["review"].astype(str).apply(preprocess_text)

# ---------------------- Train Logistic Regression Model ----------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(train_df["cleaned_review"])
y_train = train_df["sentiment"]

model_lr = LogisticRegression(max_iter=500)
model_lr.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model_lr, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Load model and vectorizer for prediction
model_lr = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------------- TF-IDF Logistic Regression Sentiment Prediction ----------------------
def predict_sentiment_tfidf(text):
    """Predict sentiment using the Logistic Regression model."""
    processed_review = preprocess_text(text)
    review_tfidf = vectorizer.transform([processed_review])
    prediction = model_lr.predict(review_tfidf)[0]
    return f"Predicted Sentiment: {prediction}"

# ---------------------- Sentiment Analysis Function ----------------------
def analyze_sentiment(text, model_choice):
    """Analyze sentiment using the selected model."""
    if model_choice == "TF-IDF Logistic Regression":
        return predict_sentiment_tfidf(text)
    elif model_choice == "TensorFlow Model":
        return predict_sentiment_tensorflow(text)
    else:
        return predict_sentiment_roberta(text)

# ---------------------- Gradio UI ----------------------
with gr.Blocks() as interface:
    gr.Markdown("# Movie Review Sentiment Analysis App")
    gr.Markdown("Enter a review, and the model will predict if it's Positive, Negative, or Neutral.")

    model_choice = gr.Dropdown(
        ["TF-IDF Logistic Regression", "TensorFlow Model", "RoBERTa Model"], 
        label="Select Model",
        value="RoBERTa Model"  # Set RoBERTa as default
    )
    text_input = gr.Textbox(label="Enter a Review", lines=5)
    output = gr.Textbox(label="Sentiment Prediction", interactive=False)

    analyze_button = gr.Button("Analyze")
    analyze_button.click(analyze_sentiment, inputs=[text_input, model_choice], outputs=output)

    # Add example inputs
    gr.Examples(
        [
            ["This movie was amazing, I loved every minute of it!", "RoBERTa Model"],
            ["This was the worst movie I've ever seen, terrible acting and plot.", "RoBERTa Model"],
            ["The movie was okay, nothing special but watchable.", "RoBERTa Model"]
        ],
        inputs=[text_input, model_choice]
    )

    # Add model comparison section
    with gr.Accordion("About the Models", open=False):
        gr.Markdown("""
        ## Model Information
        
        - **TF-IDF Logistic Regression**: A classical machine learning approach using term frequency-inverse document frequency features.
        - **TensorFlow Model**: A custom neural network trained on the training data.
        - **RoBERTa Model**: A state-of-the-art transformer model (cardiffnlp/twitter-roberta-base-sentiment-latest) from HuggingFace, fine-tuned for sentiment analysis on Twitter data.
        
        The RoBERTa model generally provides the most accurate sentiment predictions, especially for complex or nuanced text, but may be slower than the other models.
        """)

# Launch the app
interface.launch()
