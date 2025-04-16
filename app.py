import streamlit as st
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already available
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('Restaurant_Review_Model.pkl')
count_vectorizer = joblib.load('Count_Vectorizer.pkl')

def preprocess(text):
    custom_stopwords = {
        "no", "nor", "not", "isn", "isn't", "don", "don't", "ain", "ain't", "aren", "aren't", "couldn", "couldn't", 
        "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "haven", "haven't", "ma", "mightn", "mightn't", 
        "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", 
        "weren", "weren't", "won", "won't", "wouldn", "wouldn't"
    }

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english')) - custom_stopwords

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return " ".join(review)

# Streamlit UI
st.title("🍽️ Restaurant Review Sentiment Analyzer")

review = st.text_area("Enter your review here:")
if st.button("Analyze"):
    clean_review = preprocess(review)
    vectorized = count_vectorizer.transform([clean_review]).toarray()
    prediction = model.predict(vectorized)[0]
    sentiment = "👍 Positive" if prediction == 1 else "👎 Negative"
    st.subheader("Result:")
    st.success(sentiment)
