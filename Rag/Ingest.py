import os, pickle
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "Rag/index.pkl"

def chunk_text(text: str, words_per_chunk=300, overlap=50):
    words = text.split()
    chunks = []
    step = words_per_chunk - overlap
    for i in range(0, max(1, len(words)), step):
        chunk = " ".join(words[i:i+words_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def load_pdfs_to_chunks(folder="Data/Filings"):
    docs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        try:
            doc = fitz.open(path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if not text:
                    continue
                for chunk in chunk_text(text, 300, 60):
                    docs.append({
                        "text": chunk,
                        "source": fname,
                        "page": page_num + 1
                    })
        except Exception as e:
            print(f"[WARN] Failed to read {fname}: {e}")
    return docs

def build_index(folder="Data/Filings", index_path=INDEX_PATH):
    model = SentenceTransformer(EMB_MODEL_NAME)
    docs = load_pdfs_to_chunks(folder)
    if not docs:
        raise RuntimeError("No PDF chunks found. Add PDFs into data/filings and retry.")
    texts = [d["text"] for d in docs]
    embs = model.encode(texts, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)
    meta = [{"source": d["source"], "page": d["page"]} for d in docs]
    payload = {
        "model_name": EMB_MODEL_NAME,
        "embeddings": embs,
        "texts": texts,
        "meta": meta
    }
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(payload, f)
    return index_path, len(texts)

if __name__ == "__main__":
    p, n = build_index()
    print(f"Built index at {p} with {n} chunks.")
