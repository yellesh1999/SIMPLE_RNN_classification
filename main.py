import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load IMDB dataset 
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

#load pre-trained model with relu activation
model = load_model('simple_rnn_IMDB.keras')

# helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

#function to preprocess review
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## stream lit app
import streamlit as st
st.title('IMDB movie review sentiment analysis')
st.write('enter a movie review to classify it as positive or negative.')

#user input
user_input = st.text_area('movie Review')

if st.button('classify'):

    preprocess_input=preprocess_text(user_input)

    ## make prediction
    prediction= model.predict(preprocess_input)
    sentiments = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #display results
    st.write(f'Sentiment:{sentiments}')
    st.write(f'Prediction Score : {prediction[0][0]}')
else:
    st.write('please enter a movie review')
