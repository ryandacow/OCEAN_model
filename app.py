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

traits = ['Extraversion', 'Neuroticism', 'Agreeable', 'Conscientiousness', 'Openness']

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
with open("models/topic_stopwords.pkl", "rb") as f:
    topic_stopwords = pickle.load(f)

def preprocess_data(text, expand_contractions = True, use_lemmanization = True):
    if expand_contractions:
        text = contractions.fix(text)
    
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]

    stop_words = set(stopwords.words('english'))
    custom_stopwords = {
    'im', 'ive', 'id', 'youre', 'theyre', 'weve', 'hes', 'shes', 'thats', 'wasnt', 'isnt',
    'aint', 'dont', 'doesnt', 'didnt', 'couldnt', 'wouldnt', 'shouldnt',
    'wont', 'cant', 'couldve', 'wouldve', 'shouldve',
    'yea', 'yeah', 'nah', 'nope', 'ok', 'okay', 'alright', 'hey', 'hi', 'hello',
    'hmm', 'umm', 'uh', 'uhh', 'uhm', 'lol', 'lmao', 'omg', 'idk', 'ikr', 'btw',
    'pls', 'please', 'thx', 'thanks', 'thankyou', 'thank', 'like', 'just', 'really',
    'actually', 'literally', 'kinda', 'sorta', 'maybe', 'probably', 'perhaps',
    'well', 'gotta', 'gonna', 'wanna', 'lemme', 'gimme', 'cuz', 'cause', 'tho', 'tho.', 'Yaaaaay',
    'lol.', 'lmao.', 'huh', 'yo', 'sup', 'nah', 'okay', 'ok', 'oof', 'whoa', 'wow', 'ugh', 'whats', '\'s', 'oh', '``'
    }
    custom_stopwords.update({
    'people', 'think', 'get', 'know', 'time', 'want', 'good', 'way', 'see', 'something',
    'make', 'things', 'need', 'go', 'right', 'thing', 'lot', 'feel', 'sure', 'work', 
    'got', 'better', 'someone', 'life', 'said', 'find', 'first', 'many', 
    'pretty', 'back', 'take', 'person', 'years', 'long',
    'cogfuncmention', 'typemention', 'tonight', 'today'
    })
    custom_stopwords.update([
    'would', 'one', 'also', 'even', 'much',
    'could', 'still', 'say', 'going', 'though', 'use'
    ])
    # Be careful of this
    custom_stopwords.update([
    'anything', 'every', 'around', 'two', 'end', 'us', 'ill', 'since', '1', 'theres', 'etc', 'getting'
    ])
    all_stopwords = stop_words.union(custom_stopwords)
    all_stopwords.discard('love')  # Make sure "love" is retained
    clean_tokens = [token for token in tokens if token not in all_stopwords]

    clean_tokens = [token for token in clean_tokens if token not in topic_stopwords]

    name_set = set(names.words())
    name_set.discard('Love')
    clean_tokens = [token for token in clean_tokens if token.capitalize() not in name_set]

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # fallback

    if use_lemmanization:
        lemmatizer = WordNetLemmatizer()
        tagged = pos_tag(clean_tokens)
        clean_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    
    return ' '.join(clean_tokens)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    source = data.get("source", "unknown")  # Default to "unknown" if None

    clean_text = preprocess_data(text, expand_contractions=True, use_lemmanization=True)
    tfidf_vec = tfidf.transform([clean_text])
    ohe_vec = ohe.transform([[source]])

    # Combine TF-IDF and source OHE
    combined_vec = hstack([tfidf_vec, ohe_vec])

    # Calculate scores = intercept + weighted dot product
    weights = [clf.coef_[0] for clf in model.estimators_]
    scores = np.array([clf.intercept_[0] for clf in model.estimators_])
    for i, trait_weights in enumerate(weights):
        scores[i] += combined_vec.dot(trait_weights.T)[0]

    # Sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-scores))
    probas_rounded = np.round(probs, 3)
    trait_probs = dict(zip(traits, probas_rounded))

    # Margin logic: pick traits close to top prediction
    threshold = 0.4
    margin = 0.02
    top_idx = np.argmax(probs)
    top_score = probs[top_idx]
    predicted_traits = [
        traits[i] for i, p in enumerate(probs)
        if p >= threshold and (top_score - p <= margin)
    ]

    # Trait-specific thresholds (optional override)
    trait_thresholds = {
        'Extraversion': 0.4,
        'Neuroticism': 0.4,
        'Agreeable': 0.4,
        'Conscientiousness': 0.4,
        'Openness': 0.5
    }
    binary_vector = [
        1 if probs[i] >= trait_thresholds[traits[i]] else 0
        for i in range(len(traits))
    ]

    return jsonify({
        "predicted_traits": predicted_traits,
        "probabilities": trait_probs,
        "binary_vector": binary_vector
    })

if __name__ == "__main__":
    app.run(debug=True)