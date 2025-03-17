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

# Download stopwords if not available
nltk.download("stopwords")

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

        processed_text = preprocess_text(text)
        
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="pre")
        prediction = model_tnsorflow.predict(padded_sequence)[0]
        sentiment = "POSITIVE" if prediction[1] > 0.5 else "NEGATIVE"
        return sentiment
    except Exception as e:
        return f"Error in prediction: {e}"

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
    else:
        return predict_sentiment_tensorflow(text)

# ---------------------- Gradio UI ----------------------
with gr.Blocks() as interface:
    gr.Markdown("# Movie Review Sentiment Analysis App")
    gr.Markdown("Enter a review, and the model will predict if it's Positive, Negative, or Neutral.")

    model_choice = gr.Dropdown(["TF-IDF Logistic Regression", "TensorFlow Model"], label="Select Model")
    text_input = gr.Textbox(label="Enter a Review")
    output = gr.Textbox(label="Sentiment Prediction", interactive=False)

    analyze_button = gr.Button("Analyze")
    analyze_button.click(analyze_sentiment, inputs=[text_input, model_choice], outputs=output)

# Launch the app
interface.launch()
