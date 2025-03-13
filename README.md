### **Movie Review Sentiment Analysis Project - Description**  

✅ **Project Overview**  
- A **movie review sentiment analysis** system that predicts whether a review is **Positive, Negative, or Neutral**.  
- Uses **Natural Language Processing (NLP)** techniques for text preprocessing and machine learning for classification.  
- Built with **Gradio** for an interactive web-based interface.  

✅ **Key Features**  
- **Text Preprocessing**:  
  - Converts text to lowercase.  
  - Removes special characters and extra spaces.  
  - Eliminates stopwords (common words like *"the"*, *"and"*, *"is"*, etc.).  

- **Machine Learning Model**:  
  - Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features.  
  - Trained with **Logistic Regression**, a popular classification algorithm.  

- **Model Deployment with Gradio**:  
  - Provides an easy-to-use **web interface**.  
  - Users can **input a review**, and the model predicts the sentiment instantly.  

✅ **Technology Stack**  
- **Python** (Main Programming Language)  
- **Pandas** (Data Handling)  
- **NLTK** (Natural Language Processing)  
- **Scikit-learn** (Machine Learning)  
- **Gradio** (Web Interface)  
- **Joblib** (Model Saving & Loading)  

✅ **How It Works**  
1. Loads **movie reviews** from a CSV dataset.  
2. Preprocesses text (cleaning, tokenization, stopword removal).  
3. Transforms text into **TF-IDF features**.  
4. Trains a **Logistic Regression model** to classify sentiment.  
5. Saves the trained model for future predictions.  
6. Uses **Gradio** to provide a web-based interface for easy user interaction.  

✅ **Usage**  
- Run the script, enter a movie review, and get a **sentiment prediction** instantly! 🚀  
