import pandas as pd
from rag_fir_engine import retrieve_legal_sections

K = 5  # top-k retrieval

def extract_ipc_codes(texts):
    codes = []
    for t in texts:
        if t.startswith("IPC Section"):
            code = t.split(":")[0].replace("IPC Section", "").strip()
            codes.append(code)
    return codes


# 🔹 LOAD EVALUATION DATA (FROM data FOLDER)
data = pd.read_csv("data/evaluation_data.csv")

precision_scores = []
recall_scores = []

for _, row in data.iterrows():
    fir = row["fir_text"]
    expected = row["expected_ipc"].split(";")

    ipc_sections, _ = retrieve_legal_sections(fir, top_k=K)
    retrieved = extract_ipc_codes(ipc_sections)

    correct = set(expected).intersection(set(retrieved))

    precision = len(correct) / K
    recall = len(correct) / len(expected)

    precision_scores.append(precision)
    recall_scores.append(recall)

print("\n📊 RAG EVALUATION RESULTS")
print(f"Average Precision@{K}: {sum(precision_scores)/len(precision_scores):.2f}")
print(f"Average Recall@{K}: {sum(recall_scores)/len(recall_scores):.2f}")

