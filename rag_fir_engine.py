# -----------------------------
# SSL BYPASS
# -----------------------------
import os
import ssl
import urllib3

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------
# IMPORTS
# -----------------------------
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# LOAD VECTOR DATABASE
# -----------------------------
index = faiss.read_index("vector_db/index.faiss")

with open("vector_db/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# -----------------------------
# LOAD EMBEDDING MODEL (ONCE)
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# LIGHT FIR PREPROCESSING
# -----------------------------
def preprocess_fir(text: str) -> str:
    return text.lower().strip()

# -----------------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------------
def retrieve_sections(fir_text, top_k=5):
    # Normalize FIR
    fir_text = preprocess_fir(fir_text)

    # Encode query
    query_embedding = model.encode([fir_text]).astype("float32")

    # FAISS search
    distances, indices = index.search(query_embedding, top_k)

    # Normalize distances for confidence
    max_distance = float(max(distances[0])) if max(distances[0]) != 0 else 1.0

    results = []
    for i, idx in enumerate(indices[0]):
        distance = float(distances[0][i])
        confidence = 1 - (distance / max_distance)

        results.append({
            "section_text": texts[int(idx)],
            "confidence": round(float(confidence), 2)
        })

    return results
