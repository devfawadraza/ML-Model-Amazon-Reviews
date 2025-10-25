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

# def clean_text(text):
#     text = str(text).lower()
#     for c, f in contractions.items():
#         text = text.replace(c, f)
#     text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
#     text = re.sub(r"not\s+(\w+)", r"not_\1", text)          # preserve negations
#     text = re.sub(r"[^a-z\s]", "", text)        # keep only letters
#     text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # limit repeated chars
#     text = re.sub(r"[^a-z_!? ]", " ", text)                 # keep letters, _, !, ?
#     text = re.sub(r"\s+", " ", text).strip()
#     words = text.split()
#     words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
#     return " ".join(words)

@app.route("/")
def home():
    return render_template("index.html")

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