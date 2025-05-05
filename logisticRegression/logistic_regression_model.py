import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords if not available
nltk.download("stopwords")

# ---------------------- Load & Preprocess Dataset ----------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def load_and_preprocess_data(train_path, test_path=None):
    # Load training data
    train_df = pd.read_csv(train_path)
    
    # Apply text preprocessing
    train_df["cleaned_review"] = train_df["review"].astype(str).apply(preprocess_text)
    
    # Load test data if provided
    test_df = None
    if test_path:
        test_df = pd.read_csv(test_path)
        test_df["cleaned_review"] = test_df["review"].astype(str).apply(preprocess_text)
    
    return train_df, test_df

# ---------------------- Train Logistic Regression Model ----------------------
def train_logistic_regression_model(train_df, max_features=5000, ngram_range=(1, 2), max_iter=500):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(train_df["cleaned_review"])
    y_train = train_df["sentiment"]

    model_lr = LogisticRegression(max_iter=max_iter)
    model_lr.fit(X_train_tfidf, y_train)
    
    return model_lr, vectorizer

# ---------------------- TF-IDF Logistic Regression Sentiment Prediction ----------------------
def predict_sentiment_tfidf(text, model_lr=None, vectorizer=None):
    """Predict sentiment using the Logistic Regression model."""
    if model_lr is None or vectorizer is None:
        # Load model and vectorizer if not provided
        model_lr = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
    processed_review = preprocess_text(text)
    review_tfidf = vectorizer.transform([processed_review])
    prediction = model_lr.predict(review_tfidf)[0]
    return prediction

# ---------------------- Test Model Accuracy ----------------------
def test_model_accuracy(test_df, model_lr=None, vectorizer=None):
    """Test the accuracy of the model on a test dataset."""
    if model_lr is None or vectorizer is None:
        # Load model and vectorizer if not provided
        model_lr = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(test_df["cleaned_review"])
    y_test = test_df["sentiment"]
    
    # Make predictions
    y_pred = model_lr.predict(X_test_tfidf)
    
    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return accuracy, report, conf_matrix

# Example usage
if __name__ == "__main__":
    train_path = r"train_data.csv"
    test_path = r"train_data.csv"  # Using train data as test for demonstration (ideally use separate test data)
    
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data(train_path, test_path)
    
    # Train the model
    model_lr, vectorizer = train_logistic_regression_model(train_df)
    
    # Save the trained model and vectorizer
    print("Saving trained model and vectorizer...")
    joblib.dump(model_lr, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Model and vectorizer saved successfully.")
    
    # Test the model's accuracy
    if test_df is not None:
        print("Testing model accuracy...")
        accuracy, report, conf_matrix = test_model_accuracy(test_df, model_lr, vectorizer)
    
    # Example prediction
    sample_review = "This movie was absolutely fantastic! Great acting and storyline."
    prediction = predict_sentiment_tfidf(sample_review, model_lr, vectorizer)
    print(f"\nSample Review: {sample_review}")
    print(f"Predicted Sentiment: {prediction}")
