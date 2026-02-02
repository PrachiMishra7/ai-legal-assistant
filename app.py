# -----------------------------
# IMPORTS
# -----------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import RAG retrieval function
from rag_fir_engine import retrieve_sections

# -----------------------------
# FASTAPI APP INITIALIZATION
# -----------------------------
app = FastAPI(
    title="AI Legal Assistant",
    description="AI-based FIR Analysis using IPC & CrPC",
    version="1.1"
)

# -----------------------------
# CORS MIDDLEWARE
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class FIRRequest(BaseModel):
    fir_text: str

# -----------------------------
# ROOT ROUTE → LOAD UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """
    Serves the frontend UI
    """
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# FIR ANALYSIS ENDPOINT
# -----------------------------
@app.post("/analyze_fir")
def analyze_fir(req: FIRRequest):
    """
    Analyzes FIR text and returns:
    - Dynamic legal summary
    - Relevant IPC & CrPC sections
    - Section distribution counts
    - Precision@K, Recall@K, Accuracy (demo metric)
    """

    # Retrieve relevant legal sections
    sections = retrieve_sections(req.fir_text)

    # Separate IPC and CrPC section titles
    ipc_sections = [
        s["section_text"].split("\n")[0]
        for s in sections
        if "IPC Section" in s["section_text"]
    ]

    crpc_sections = [
        s["section_text"].split("\n")[0]
        for s in sections
        if "CrPC Section" in s["section_text"]
    ]

    # -----------------------------
    # DYNAMIC SUMMARY GENERATION
    # -----------------------------
    if ipc_sections:
        summary = (
            "Based on the contents of the FIR, the alleged incident appears to involve "
            + ", ".join(ipc_sections[:2])
            + ". These provisions indicate the presence of cognizable criminal acts. "
        )
    else:
        summary = (
            "Based on the FIR description, no specific IPC offences could be conclusively identified. "
        )

    if crpc_sections:
        summary += (
            "Relevant procedural provisions such as "
            + ", ".join(crpc_sections[:2])
            + " may apply during the investigation process."
        )

    # -----------------------------
    # PERFORMANCE METRICS
    # -----------------------------
    k = 5  # top-k retrieval size

    # Precision@K = relevant retrieved / total retrieved
    precision_at_k = round(len(ipc_sections) / k, 2) if sections else 0.0

    # Recall@K = relevant retrieved / expected relevant (assume 2 for demo)
    recall_at_k = round(min(len(ipc_sections), 2) / 2, 2) if ipc_sections else 0.0

    # Accuracy-style score (for visualization/demo only)
    accuracy = round((precision_at_k + recall_at_k) / 2, 2)

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return {
        "summary": summary,
        "sections": sections,
        "ipc_count": int(len(ipc_sections)),
        "crpc_count": int(len(crpc_sections)),

        # 🔥 NEW METRICS
        "precision": precision_at_k,
        "recall": recall_at_k,
        "accuracy": accuracy
    }
