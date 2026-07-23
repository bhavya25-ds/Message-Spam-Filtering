import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_resource

def train_model():
    df= pd.read_csv("spam.csv", encoding= "latin-1")
    df['Class']= df['Class'].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test= train_test_split(df['Message'], df["Class"], test_size=0.2, random_state=42, stratify=df["Class"])

    vectorizer= TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1,2))
    X_train_vec= vectorizer.fit_transform(X_train)
    X_test_vec= vectorizer.transform(X_test)

    model= ComplementNB()
    model.fit(X_train_vec,y_train)

    accuracy_ts= accuracy_score(y_test, model.predict(X_test_vec))

    X_all_vec= vectorizer.fit_transform(df["Message"])
    model.fit(X_all_vec, df['Class'])

    return vectorizer, model, accuracy_ts

vectorizer, model, accuracy_ts= train_model()

st.title("📩 SMS Spam Classifier")
st.subheader("Classify your day to day text messages as spam or ham!")
st.caption(f"Model accuracy on held-out test data: {accuracy_ts * 100:.1f}%")
st.write("Enter the message you want to classify below 👇")
msg= st.text_area("Enter your message: ", height= 150)

if st.button("Check"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed= vectorizer.transform([msg])
        prediction= model.predict(transformed)[0]
        prob= model.predict_proba(transformed)[0]

        if prediction == 1:
            st.error(f"🚨 SPAM- {round(prob[1] * 100, 1)}% confidence")
        else:
            st.success(f"✅ HAM- {round(prob[0] * 100, 1)}% confidence")
