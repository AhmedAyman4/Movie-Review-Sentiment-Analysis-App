FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords at build time
RUN python -c "import nltk; nltk.download('stopwords')"

# Copy all application files
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p templates static

# Expose port for the Flask application
EXPOSE 5000

# Use gunicorn as the production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
