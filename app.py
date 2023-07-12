import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

import os
import nltk

nltk.download('punkt')
nltk.download('stopwords')


# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath('app.py'))

# Construct the absolute path to the pickled file
file_path = os.path.join(script_dir, 'vectorizer.pkl')

# Load the pickled object
tfidf = pickle.load(open(file_path, 'rb'))

model=pickle.load(open("model.pkl",'rb'))

st.title("SMS spam classifier")
text = st.text_area("Enter the message")


def preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)
    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(stemmer.stem(i))

    return " ".join(y)

if st.button("Predict"):

    transformed= preprocess(text)

    vector = tfidf.transform([transformed])

    result = model.predict(vector)[0]


    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")