from transformers import pipeline

# Load once; Streamlit will cache at call site
def load_finbert_pipeline():
    return pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=False)

def classify_sentiment(nlp, text: str):
    out = nlp(text)[0]  # {'label': 'positive'/'negative'/'neutral', 'score': ...}
    return out["label"], float(out["score"])
