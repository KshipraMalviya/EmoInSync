import base64
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(page_title='EmoInSync', page_icon = "assets/icon.png")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
    }
    [data-testid="stHeader"] {
        background-color: white;
    }
    [data-testid="baseButton-secondary"] {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

lg=pickle.load(open('logistic_regression.pkl','rb'))
lb=pickle.load(open('label_encoder.pkl','rb'))
tfidf_vectorizer=pickle.load(open('tfidf_vectorizer.pkl','rb'))

stopwords_set = set(stopwords.words('english'))

def clean_text(text):
    stemmer=PorterStemmer()
    text=text.lower()
    text=re.sub("[^a-z]"," ",text)
    text=text.split()
    text=[stemmer.stem(word) for word in text if word not in stopwords_set]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text=clean_text(input_text)
    input_vector=tfidf_vectorizer.transform([cleaned_text])
    predicted_label=lg.predict(input_vector)[0]
    predicted_emotion=lb.inverse_transform([predicted_label])[0]
    label=np.max(predicted_label)
    return predicted_emotion,label

st.title("EmoInSync - 6 Human Emotions Recognizer✨")

input=st.text_area("Your text here:")
if st.button("Predict"):
    if len(input):
        predicted_emotion,label=predict_emotion(input)
        st.write("Predicted emotion: ",predicted_emotion)
        file_ = open(f"assets/{predicted_emotion}.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" width="320px">',
            unsafe_allow_html=True,
        )
    else:
        st.write("Please enter some text!")