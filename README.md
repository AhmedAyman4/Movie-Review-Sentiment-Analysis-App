# Movie Review Sentiment Analysis Flask App

This Flask application serves a sentiment analysis service for movie reviews using three different models:
1. TF-IDF Logistic Regression
2. Custom TensorFlow Neural Network
3. HuggingFace RoBERTa Transformer Model

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

## Further Improvements

- Add user authentication
- Implement rate limiting
- Add batch processing capabilities
- Store analysis results in a database
- Add caching for better performance
- Create API documentation with Swagger/OpenAPI
