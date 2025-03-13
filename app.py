import gradio as gr
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.model_selection import train_test_split
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Download stopwords if not available
nltk.download('stopwords')
custom_stopwords = set(stopwords.words('english')) 
# Paths to datasets
train_path = r"E:\Projects\Sentiment Analysis Project DEPI\train_data.csv"
test_path = r"E:\Projects\Sentiment Analysis Project DEPI\test_data.csv"

# Load datasets
train_df = pd.read_csv(train_path)

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):  # Ensure text is a string
        ## REMOVE HTML
        if "<" in text and ">" in text:
            text = BeautifulSoup(text, "html.parser").get_text()
        ## CLEANING
        # Remove special characters
        text = re.sub(r'\W+', ' ', text)  
        # Remove digits
        text = re.sub(r'\d+', '', text)
        ## LOWERCASING
        text = text.lower()
        ## TOKENIZATION
        words = text.split()
        ## REMOVE STOPWORDS
        words = [w for w in words if w not in custom_stopwords] 
        ## APPLY LEMMATIZATION
        words = [lemmatizer.lemmatize(w) for w in words]
        ## RETURN CLEANED TEXT
        return ' '.join(words)
    return ""

# Apply preprocessing
train_df['cleaned_review'] = train_df['review'].astype(str).apply(preprocess_text)

# Train the model
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(train_df['cleaned_review'])
y_train = train_df['sentiment']

model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Load model and vectorizer for prediction
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Gradio prediction function
def predict_sentiment(review):
    processed_review = preprocess_text(review)  # Preprocess input
    review_tfidf = vectorizer.transform([processed_review])  # Convert to TF-IDF
    prediction = model.predict(review_tfidf)[0]  # Get prediction
    return f"Predicted Sentiment: {prediction}"

# Gradio UI
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter a Review"),
    outputs=gr.Textbox(label="Sentiment Prediction"),
    title="Movie Review Sentiment Analysis App",
    description="Enter a review, and the model will predict if it's Positive, Negative, or Neutral."
)

# Launch the app
interface.launch()
