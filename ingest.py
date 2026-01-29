import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import ssl

# ----------------------------------
# SSL FIX (EXACTLY AS YOU GAVE)
# ----------------------------------
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

# ----------------------------------
# LOAD DATASETS (ENCODING FIX)
# ----------------------------------
ipc = pd.read_csv("data/ipc_sections.csv", encoding="latin1")
crpc = pd.read_csv("data/crpc_sections.csv", encoding="latin1")

texts = []

# ----------------------------------
# IPC DATA PROCESSING
# ----------------------------------
for _, row in ipc.iterrows():
    text = f"""
IPC Section {row['Section']}:
Offence: {row['Offense']}
Description: {row['Description']}
Punishment: {row['Punishment']}
"""
    texts.append(text.strip())

# ----------------------------------
# CRPC DATA PROCESSING
# ----------------------------------
for _, row in crpc.iterrows():
    text = f"""
CrPC Section {row['Section']}:
Section Name: {row['Section _name']}
Description: {row['Description']}
"""
    texts.append(text.strip())

# ----------------------------------
# LOAD EMBEDDING MODEL
# ----------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------
# CONVERT TEXT TO EMBEDDINGS
# ----------------------------------
embeddings = model.encode(texts, show_progress_bar=True)

# ----------------------------------
# CREATE FAISS INDEX
# ----------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ----------------------------------
# SAVE VECTOR DATABASE
# ----------------------------------
os.makedirs("vector_db", exist_ok=True)
faiss.write_index(index, "vector_db/index.faiss")

with open("vector_db/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ Vector database created successfully")
print(f"📚 Total legal sections indexed: {len(texts)}")
