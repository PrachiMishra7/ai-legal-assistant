from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_fir_engine import retrieve_sections

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FIRRequest(BaseModel):
    fir_text: str

# 🔥 DEFAULT ROUTE OPENS UI
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analyze_fir")
def analyze_fir(req: FIRRequest):
    sections = retrieve_sections(req.fir_text)

    summary = (
        "The FIR describes an alleged criminal incident. "
        "Based on semantic analysis, the system identified "
        "relevant IPC and CrPC sections applicable to the case."
    )

    ipc_count = len([s for s in sections if "IPC Section" in s["section_text"]])
    crpc_count = len([s for s in sections if "CrPC Section" in s["section_text"]])

    return {
        "summary": summary,
        "sections": sections,
        "ipc_count": ipc_count,
        "crpc_count": crpc_count
    }
