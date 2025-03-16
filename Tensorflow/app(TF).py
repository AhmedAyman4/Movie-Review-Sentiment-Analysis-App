#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# In[2]:


try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model("best_model.keras")
    max_len = 1128
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    tokenizer = None
    model = None


# In[4]:


def predict_sentiment(text):
    try:
        if not tokenizer or not model:
            return "Error: Model or Tokenizer not loaded properly."
        
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='pre')
        prediction = model.predict(padded_sequence)[0]
        sentiment = "POSITIVE" if prediction[1] > 0.5 else "NEGATIVE"
        return sentiment  # لازم تبقى جوة الـ try block
    except Exception as e:
        return f"Error in prediction: {e}"


# In[8]:


interface = gr.Interface(fn=predict_sentiment, inputs="text", outputs="text")


# In[10]:


try:
    interface.launch()
except Exception as e:
    print(f"Error launching Gradio interface: {e}")


# In[ ]:




