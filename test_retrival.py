import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ---------------------------
# FORCE OFFLINE MODE (IMPORTANT)
# ---------------------------
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------
# LOAD VECTOR DATABASE
# ---------------------------
index = faiss.read_index("vector_db/index.faiss")

with open("vector_db/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# ---------------------------
# LOAD EMBEDDING MODEL (LOCAL ONLY)
# ---------------------------
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

# ---------------------------
# SAMPLE FIR-LIKE QUERY
# ---------------------------
query = """
The accused assaulted the complainant and threatened him with serious consequences.
"""

query_embedding = model.encode([query])

# ---------------------------
# SEARCH TOP K SECTIONS
# ---------------------------
k = 5
_, indices = index.search(query_embedding, k)

# ---------------------------
# DISPLAY RESULTS
# ---------------------------
print("\n🔍 RETRIEVED LEGAL SECTIONS:\n")

for i in indices[0]:
    print("--------------------------------------------------")
    print(texts[i])
