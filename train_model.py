# model training script k liye
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
import string

# =============Download NLP tools for gaining 90-95% accuracy=================

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ================Load dataset====================
df = pd.read_csv("data/amazon_reviews.csv", engine="python", on_bad_lines="skip", encoding="utf-8")
print("Dataset Loaded: ",df.shape)

# df = df.sample(100000, random_state=42)
# print(f"Using sample of: {df.shape}")


 # ============Drop Null values====================
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)

 # ============keep numeric labels=========
df['label'] = df['label'].astype(int)
df = df[df['label'].isin([1, 2])]



# 👇 Add this to inspect your text column before cleaning
# print("Sample text values before cleaning:")
# print(df['text'].head(10))


# =============clean the text=====================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

contractions = {
    "don't": "do not",
    "can't": "cannot",
    "i'm": "i am",
    "it's": "it is",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not"
}

def clean_text(text):
    text = str(text).lower()
    for c, f in contractions.items():
        text = text.replace(c, f)
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"not\s+(\w+)", r"not_\1", text)          # preserve negations
    text = re.sub(r"[^a-z\s]", "", text)        # keep only letters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # limit repeated chars
    text = re.sub(r"[^a-z_!? ]", " ", text)                 # keep letters, _, !, ?
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned_text'] = df['text'].apply(clean_text)
print("Text cleaned....")


# ================split the text==================
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42 
)

# ================Vectorize text==================
print("Vectorizing the text")
start_time = time.time()
vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 3),       # use unigrams + bigrams # i used trigrams after uni & bigrams
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
    smooth_idf=True,
    stop_words = stopwords.words('english')
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Vectorization completed in {time.time() - start_time:.2f} seconds.")

# ============Logistic Regression hyperparameter tuning=================
param_grid = {
    'C': [1.0, 2.0, 3.0],
    'solver': ['saga'],
    'penalty': ['l2'],
    'max_iter': [800]
}

grid = GridSearchCV(
    LogisticRegression(n_jobs=-1, class_weight='balanced'),
    param_grid,
    scoring= 'accuracy',
    cv= 3,
    verbose=2
)

print("Starting Hyperparameter tuning...")
grid.fit(X_train_vec, y_train)
print("Best Hyperparameter:", grid.best_params_)


# =============train final model============
model = grid.best_estimator_
model.fit(X_train_vec, y_train)
print("Model training complete")

# =================Evaluate=======================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ===========Save model and vectorizer============
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("Model & Vectorizer saved successfully!!!")