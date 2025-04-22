from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import string
import contractions
import nltk
from nltk.corpus import stopwords, names, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from scipy.sparse import hstack

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("names", quiet=True)

# Load models
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("models/ohe_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)
with open("models/traits_list.pkl", "rb") as f:
    traits = pickle.load(f)

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("names")

# Load names and stopwords
stop_words = set(stopwords.words("english"))
name_set = set(names.words())
lemmatizer = WordNetLemmatizer()

# Sample topic stopwords (you can also load these from a .pkl if needed)
topic_stopwords = {"tonight", "today"}  # minimal fallback if not saved

def preprocess_data(text):
    text = contractions.fix(text)
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]

    custom_stopwords = {'im', 'ive', 'id', 'youre', 'theyre', 'ok', 'okay', 'like', 'just', 'really'}
    all_stopwords = stop_words.union(custom_stopwords)
    all_stopwords.discard("love")

    tokens = [t for t in tokens if t not in all_stopwords and t not in topic_stopwords]
    tokens = [t for t in tokens if t.capitalize() not in name_set]

    def get_wordnet_pos(tag):
        if tag.startswith("J"): return wordnet.ADJ
        elif tag.startswith("V"): return wordnet.VERB
        elif tag.startswith("N"): return wordnet.NOUN
        elif tag.startswith("R"): return wordnet.ADV
        return wordnet.NOUN

    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    return " ".join(lemmatized)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    source = None

    clean_text = preprocess_data(text)
    tfidf_vec = tfidf.transform([clean_text])
    ohe_vec = ohe.transform([[source]])

    # Combine both vectors
    combined_vec = hstack([tfidf_vec, ohe_vec])
    probs = [clf.predict_proba(combined_vec)[0][1] for clf in model.estimators_]

    # Format response
    trait_probs = dict(zip(traits, np.round(probs, 3)))
    predicted_traits = [traits[i] for i, p in enumerate(probs) if p >= 0.4]

    return jsonify({
        "predicted_traits": predicted_traits,
        "probabilities": trait_probs
    })

if __name__ == "__main__":
    app.run(debug=True)