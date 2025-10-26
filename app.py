# Flask backend k liye

from flask import Flask, render_template, request, jsonify
import joblib
import re


app = Flask(__name__)

#Load model and vectorizer

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def clean_text(text):
    text = re.sub(r"https\S+", "", text)
    text = re.sub(r"a-zA-Z\s", "", text)
    text = text.lower().strip()
    return text

# @app.route("/")
# def home():
#     return render_template("index.html")

@app.route("/predict", methods=["POST"])
def perdict():
    data = request.get_json()
    text = clean_text(data['review'])
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    sentiment = "Positive" if prediction == 2 else "Negative"
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)