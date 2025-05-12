# Movie Review Sentiment Analysis Flask App

This Flask application serves a sentiment analysis service for movie reviews using three different models:
1. TF-IDF Logistic Regression
2. Custom TensorFlow Neural Network
3. HuggingFace RoBERTa Transformer Model

# Movie Review Sentiment Analysis Gradio Dash
In addition to the Flask interface, the app is now also available via:
- ✅ **Gradio UI on Hugging Face Spaces**
- ✅ **Dash Interactive Dashboard on Hugging Face Spaces**

You can **select any of the three models** in the web or API interface.
link [ https://huggingface.co/spaces/ahmed-ayman/Sentiment-Analysis ]


## Project Structure

```
sentiment-analysis-flask/
│
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── static/              
│   └── style.css         # Custom CSS styles
├── templates/
│   └── index.html        # HTML template for the web interface
├── sentiment_model.pkl   # TF-IDF Logistic Regression model (saved with joblib)
├── tfidf_vectorizer.pkl  # TF-IDF vectorizer (saved with joblib)
├── best_model.keras      # TensorFlow model
└── tokenizer.pkl         # Tokenizer for TensorFlow model
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Files

Ensure the following model files are in the same directory as app.py:
- sentiment_model.pkl
- tfidf_vectorizer.pkl
- best_model.keras
- tokenizer.pkl

## 🧠 TensorFlow Neural Network

A simple and efficient sentiment classification model built with Keras.

### Model Overview
- **Layers**: `Embedding` → `SpatialDropout1D` → `GlobalAveragePooling1D` → `Dropout` → `Dense (softmax)`
- **Loss**: `categorical_crossentropy`
- **Optimizer**: `Adam (lr=0.0005)`
- **Callbacks**: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- **Validation Accuracy**: ~90%
- **Model File**: `best_model.keras`

This model is available for selection through the web interface.

## 🧮 TF-IDF + Logistic Regression

A traditional machine learning model that uses term frequency–inverse document frequency (TF-IDF) to convert text into numerical features, followed by Logistic Regression for classification.

### Model Overview
- **Vectorizer**: `TF-IDF` (`sklearn.feature_extraction.text.TfidfVectorizer`)
- **Classifier**: `LogisticRegression`
- **Accuracy**: ~88% on validation data
- **Model Files**:
  - `sentiment_model.pkl`
  - `tfidf_vectorizer.pkl`

This model is fast and lightweight, suitable for quick inference.

## 🤖 RoBERTa Transformer (Hugging Face)

A state-of-the-art transformer-based model for sentiment analysis using Hugging Face's `transformers` library.

### Model Overview
- **Base Model**: `roberta-base` from Hugging Face
- **Tokenization & Inference**: `AutoTokenizer`, `AutoModelForSequenceClassification`
- **Accuracy**: ~92% on test data (using pretrained weights)
- **Note**: The model is automatically downloaded at runtime from Hugging Face Hub.

This model offers the best performance and understanding of context, recommended for high-quality predictions.


Note: The RoBERTa model will be downloaded from HuggingFace when running the app for the first time.

### 3. Directory Structure

Create the necessary directories:
```bash
mkdir -p templates static
```

### 4. Running the App

```bash
# Run the Flask app
python app.py
```

The application will be available at http://localhost:5000

## API Endpoints

### Web Interface
- `GET /`: Main web interface

### Analysis Endpoint
- `POST /analyze`: Analyze sentiment of a review
  - Request body:
    ```json
    {
        "review": "Your review text here",
        "model": "RoBERTa Model" 
    }
    ```
  - Model options: "RoBERTa Model", "TensorFlow Model", "TF-IDF Logistic Regression"
  - Response example:
    ```json
    {
        "sentiment": "POSITIVE",
        "confidence": 0.9876,
        "message": "Predicted Sentiment: POSITIVE (Confidence: 0.9876)"
    }
    ```

### Health Check
- `GET /health`: Check API and models status

## Deployment

For production deployment:

1. Use a production WSGI server:
```bash
pip install gunicorn
gunicorn app:app
```

2. Set debug mode to False in app.py:
```python
app.run(host="0.0.0.0", port=5000, debug=False)
```

3. Consider using Docker for containerized deployment.

## Docker Deployment

### Dockerfile

Create a Dockerfile in the project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download NLTK stopwords at build time
RUN python -c "import nltk; nltk.download('stopwords')"

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Build and Run

```bash
# Build the Docker image
docker build -t sentiment-analysis-flask .

# Run the container
docker run -p 5000:5000 sentiment-analysis-flask
```

The API endpoints are:

GET /: Web interface for sentiment analysis
POST /analyze: API endpoint for sentiment analysis
GET /health: Health check endpoint
