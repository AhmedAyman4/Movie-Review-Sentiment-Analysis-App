# Movie Sentiment Analysis Application

## Overview
This project is a comprehensive movie review sentiment analysis application that leverages multiple machine learning approaches to analyze and predict sentiment from text reviews. The application offers three different models for sentiment analysis, allowing users to compare different approaches and select the most suitable one for their needs.

## Features
- **Multiple Model Options**: Choose between three different sentiment analysis models:
  - TF-IDF with Logistic Regression (traditional ML)
  - Custom TensorFlow neural network
  - Pre-trained RoBERTa transformer model from HuggingFace
- **User-friendly Interface**: Interactive web application built with Gradio
- **Example Reviews**: Pre-loaded example reviews for quick testing
- **Confidence Scores**: View confidence scores with the RoBERTa model predictions
- **Comparative Analysis**: Information about each model's strengths and use cases

## Models

### TF-IDF with Logistic Regression
A classical machine learning approach that uses term frequency-inverse document frequency vectorization to transform text reviews into numerical features, which are then fed into a logistic regression classifier. This model is lightweight and offers fast inference.

### TensorFlow Neural Network
A custom neural network model trained on the provided movie review dataset. This model balances performance and accuracy, representing a middle ground between the simpler logistic regression and the more complex transformer model.

### RoBERTa Transformer Model
A state-of-the-art transformer model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) fine-tuned for sentiment analysis. This pre-trained model from HuggingFace offers the most nuanced and accurate predictions, especially for complex language patterns and edge cases, but may be slower than the other options.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/movie-sentiment-analysis.git
   cd movie-sentiment-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the necessary NLTK data:
   ```python
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. Ensure you have the model files in the correct location:
   - `tokenizer.pkl`: Tokenizer for the TensorFlow model
   - `best_model.keras`: The trained TensorFlow model
   - `sentiment_model.pkl`: The trained Logistic Regression model
   - `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer

   Note: The RoBERTa model will be downloaded automatically from HuggingFace on first run.

5. Prepare your training data:
   - Ensure you have a `train_data.csv` file with columns for 'review' and 'sentiment'

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860).

3. Enter a movie review in the text box, select your preferred model, and click "Analyze" to get the sentiment prediction.

## Example

```python
# Sample input
"This movie was absolutely fantastic. The performances were stellar, and the screenplay was engaging from start to finish. One of the best films I've seen this year!"

# Expected output with RoBERTa model
"Predicted Sentiment: POSITIVE (Confidence: 0.9978)"
```

## Project Structure

```
movie-sentiment-analysis/
├── app.py                   # Main application file
├── requirements.txt         # Dependencies
├── train_data.csv           # Training data
├── tokenizer.pkl            # TensorFlow tokenizer
├── best_model.keras         # TensorFlow model
├── sentiment_model.pkl      # Logistic Regression model
├── tfidf_vectorizer.pkl     # TF-IDF vectorizer
└── README.md                # Project documentation
```

## Requirements

The following libraries are required to run this application:

- gradio>=3.9.1
- pandas>=1.3.5
- numpy>=1.21.0
- nltk>=3.6.5
- joblib>=1.1.0
- scikit-learn>=1.0.2
- tensorflow>=2.8.0
- torch>=1.10.0
- transformers>=4.15.0

## Future Improvements

- Add support for batch processing of reviews
- Implement model performance metrics and comparison visualizations
- Add multi-language support for analyzing reviews in different languages
- Create a deployable API version for integration with other applications
- Implement an ensemble method combining predictions from all models

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
