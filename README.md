# Movie Review Sentiment Analysis App

## Overview
This is a machine learning application that performs sentiment analysis on movie reviews using two different models: a TF-IDF Logistic Regression model and a TensorFlow Neural Network model. The application provides a user-friendly Gradio interface for easy sentiment prediction.

## Features
- Two sentiment analysis models
- Text preprocessing
- Interactive web interface
- Supports both TF-IDF and Deep Learning approaches

## Prerequisites
- Python 3.8+
- Required Libraries:
  - gradio
  - pandas
  - numpy
  - nltk
  - scikit-learn
  - tensorflow
  - joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AhmedAyman4/Movie-Review-Sentiment-Analysis-App.git
cd Movie-Review-Sentiment-Analysis-App
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Project Structure
- `app.py`: Main application script
- `train_data.csv`: Training dataset
- `sentiment_model.pkl`: Saved Logistic Regression model
- `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer
- `best_model.keras`: Saved TensorFlow neural network model
- `tokenizer.pkl`: Saved tokenizer for TensorFlow model

## Models
### 1. TF-IDF Logistic Regression
- Uses TF-IDF vectorization
- Logistic Regression classifier
- Faster and more interpretable

### 2. TensorFlow Neural Network
- Deep learning approach
- More complex model
- Potentially better at capturing nuanced sentiments

## Text Preprocessing
- Converts text to lowercase
- Removes special characters
- Removes stopwords
- Removes extra whitespaces

## Usage
Run the application:
```bash
python app.py
```

The Gradio interface will launch, allowing you to:
1. Select a model (TF-IDF or TensorFlow)
2. Enter a movie review
3. Get sentiment prediction

## Model Performance
- Model accuracy and performance may vary
- Trained on a specific movie review dataset
- Recommended to evaluate on your specific use case

## Limitations
- Sentiment is binary (Positive/Negative)
- Performance depends on training data
- May not capture extremely nuanced sentiments

## Acknowledgments
- NLTK for text preprocessing
- Scikit-learn for machine learning tools
- TensorFlow for deep learning model
- Gradio for interactive UI
