# 🧠 OCEAN Personality Trait Predictor

This project predicts personality traits based on free-form text using the **OCEAN model** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism). It includes:

- A **Flask API** that serves predictions
- A **Jupyter notebook** with training logic and preprocessing
- A **public dataset** for transparency
- A **live demo** on my portfolio site

---

## 🌐 Try the Live Demo

Test it here:  
👉 [ryanneo.vercel.app/personality](https://ryanneo.vercel.app/personality)

---

## 🚀 Project Structure

```bash
project-root/
├── models/                # Trained model and vectorizers
│   ├── ocean_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── source_ohe.pkl
│   └── topic_stopwords.pkl
├── datasets/              # Training dataset
│   └── OCEAN_essays.csv
├── notebooks/             # Training and experimentation
│   └── OCEAN_Model.ipynb
├── app.py                 # Flask API logic
├── requirements.txt       # Dependencies for API
├── .gitignore
├── render.yaml
└── README.md
```

---

## 📊 Dataset

The dataset used for training is included in `/datasets/OCEAN_essays.csv`.  
It contains free-text essays labeled with personality traits (`y/n`) across the five OCEAN dimensions.

Additional training data is loaded in the notebook from public datasets via Hugging Face.

---

## 🧠 Model Training

All training logic is implemented in [`notebooks/OCEAN_Model.ipynb`](notebooks/OCEAN_Model.ipynb).

Training steps include:

- Text preprocessing:
  - Tokenization
  - Lowercasing
  - Stopword removal (including custom stopwords)
  - Lemmatization using NLTK
- Feature engineering:
  - TF-IDF vectorization of essay text
  - One-hot encoding of essay source
- Multi-label classification using `OneVsRestClassifier` with logistic regression

The trained model and preprocessing objects are saved as `.pkl` files in `/models/`.

---

## 🔧 Flask API

The API is implemented in [`app.py`](app.py). It:

- Accepts POST requests at `/analyze`
- Preprocesses user text using the same pipeline used in training
- Outputs:
  - Rounded probabilities for each trait
  - Binary vector of trait presence
  - Final predicted traits based on thresholds and margin logic

---

### 🔌 Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API:

```bash
python app.py
```

3. Example request:

Send a POST request to `http://localhost:5000/analyze` with a JSON body:

```json
{
  "text": "I love working with people and sharing new ideas.",
  "source": "reddit"
}
```

---

## ⚙️ Dependencies

Only the serving dependencies are included in `requirements.txt` (Flask, NumPy, NLTK, etc.).

If you want to run the training notebook, you may need to install additional packages like:

- pandas
- seaborn
- sentence-transformers
- datasets

These are intentionally excluded from the runtime environment for the API to keep it lightweight.

---

## 📫 Contact

Feel free to reach out via [my GitHub profile](https://github.com/ryandacow) with questions or feedback.

---

## 🛡 License

This project is intended for educational and research use. Attribution is appreciated.
