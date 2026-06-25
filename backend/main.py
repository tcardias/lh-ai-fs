from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from orchestrator import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def load_documents() -> dict[str, str]:
    documents = {}
    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        documents[file_path.stem] = file_path.read_text()
    return documents


@app.post("/analyze")
async def analyze():
    documents = load_documents()
    if "motion_for_summary_judgment" not in documents:
        raise HTTPException(status_code=500, detail="MSJ document not found")
    report = await run_pipeline(documents)
    return {"report": report.model_dump()}
