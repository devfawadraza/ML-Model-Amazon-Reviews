#======================Streamlit App==================
import streamlit as st
import joblib 
import re
import numpy as np

# ===============Load Model & vectorizer ======================
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")\

# ===============Text Cleaing function=========================

def clean_text(text):
    text = re.sub(r"https\S+", "", text)
    text = re.sub(r"[^a-zA-z\s]", "", text)
    text = text.lower().strip()
    return text

# ====================Streamlit Configuration==================

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis App")
st.markdown("Analyze the sentiment of the input using a trained Logisitc Regression model.")

# ======================= User Input===========================

review = st.text_area("Enter the text here")


# ================= Prediction Section ========================

if st.button("Analyze Statement"):
    if not review.strip():
        st.warning("Please enter some text first.")
    else:
        # clean + vectorize
        cleaned_text = clean_text(review)
        text_vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]

        #Map sentiment label
        sentiment = "Positive" if prediction == 2 else "Negative"
        confidence = np.max(probabilities) * 100

        # Display Result + Progress Bar
        if sentiment == "Positive":
            st.success(f"The statment is **Positive** ({confidence:.2f}% confidence)")
            st.progress(confidence / 100)
        else:
            st.error(f"The statment is **Negative** ({confidence:.2f}% confidence)")
            st.progress(confidence / 100)

        #Show both probailities and side by side

        st.write("Prediction Probalilities")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Negative", value=f"{probabilities[0]*100:.2f}%")
        with col2:
            st.metric(label="Positive", value=f"{probabilities[1]*100:.2f}%")

#==================== Custom Styling =============================

st.markdown("""
        <style>
        textarea {
            background-color: #f9f9f9;
            border-radius: 10x;
            font-size: 16px !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 16px;  
        }    
        </style>



""", unsafe_allow_html=True)