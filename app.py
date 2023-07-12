import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

tfidf= pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open("model.pkl",'rb'))

st.title("SMS spam classifier")
sms = st.text_area("Enter the message")


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

    transformed= preprocess(sms)

    vector = tfidf.transform([transformed])

    result = model.predict(vector)[0]


    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")