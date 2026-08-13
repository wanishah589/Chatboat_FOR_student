import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
VECTOR_STORE_PATH = BASE_DIR / "chroma_db" / "vectors.json"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")

EMBEDDING_MODEL = "models/gemini-embedding-001"  # current free-tier embedding model
LLM_MODEL = "gemini-3.1-flash-lite"  # current GA free-tier model (Aug 2026)
# Note: gemini-1.5-flash, gemini-2.0-flash, and gemini-2.5-flash-lite have all
# been deprecated or restricted to existing users only as of mid-2026. If this
# stops working again, check aistudio.google.com's model list under your
# project for the current free-tier model name before swapping this.

# Categories we expect inside each university's folder.
# File name (without .pdf) should match one of these, e.g. NUST/eligibility.pdf
CATEGORIES = [
    "eligibility",
    "application_process",
    "marks_requirement",
    "test_type",
    "merit_criteria",
    "scholarships_merit",
    "scholarships_need",
    "fee_structure",
    "important_dates",
    "programs_offered",
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVAL_K = 5  # how many chunks to pull per query
