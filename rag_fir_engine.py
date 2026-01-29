# -----------------------------
# SSL BYPASS
# -----------------------------
import os, ssl, urllib3
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# LOAD VECTOR DB
index = faiss.read_index("vector_db/index.faiss")

with open("vector_db/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# LOAD MODEL
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------------
def retrieve_sections(fir_text, top_k=5):
    query_embedding = model.encode([fir_text]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        confidence = 1 / (1 + float(distances[0][i]))  # 🔥 CAST TO PYTHON FLOAT

        results.append({
            "section_text": texts[int(idx)],            # 🔥 ensure int
            "confidence": round(float(confidence), 2)   # 🔥 ensure float
        })

    return results
