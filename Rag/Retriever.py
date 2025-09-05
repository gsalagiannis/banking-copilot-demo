import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index(index_path="Rag/index.pkl"):
    with open(index_path, "rb") as f:
        return pickle.load(f)

def cosine_topk(query: str, idx, k=3, min_score: float | None = None):
    """
    Return top-k results by cosine similarity.
    If min_score is set, filter out results below this threshold.
    """
    model = SentenceTransformer(idx["model_name"])
    qv = model.encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype=np.float32)  # shape (1, d)

    # cosine similarity = dot product because vectors are L2-normalized
    sims = (idx["embeddings"] @ qv.T).ravel()

    order = np.argsort(-sims)
    results = []
    for i in order[:k]:
        score = float(sims[i])
        if (min_score is not None) and (score < min_score):
            continue
        results.append({
            "score": score,
            "text": idx["texts"][i],
            "source": idx["meta"][i]["source"],
            "page": idx["meta"][i]["page"]
        })
    return results

