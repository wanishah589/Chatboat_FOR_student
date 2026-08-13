"""
Run this whenever you add or update PDFs.

Expected folder layout:
  data/pdfs/<UniversityName>/eligibility.pdf
  data/pdfs/<UniversityName>/test_type.pdf
  data/pdfs/<UniversityName>/fees.pdf
  data/pdfs/<UniversityName>/deadlines.pdf
  data/pdfs/<UniversityName>/programs.pdf

Any file name is accepted, but naming it after a category (see config.CATEGORIES)
lets the chatbot filter answers accurately (e.g. "what test does NUST require"
pulls only test_type chunks tagged university=NUST).

Usage:
  python -m app.ingest
"""

import time
from google.api_core.exceptions import ResourceExhausted
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import (
    PDF_DIR, VECTOR_STORE_PATH, GOOGLE_API_KEY, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, CATEGORIES,
)
from app.vectorstore import SimpleVectorStore


def guess_category(filename: str) -> str:
    stem = filename.lower().replace(".pdf", "")
    for cat in CATEGORIES:
        if cat in stem:
            return cat
    return "general"


def load_and_tag_documents():
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"No data folder found at {PDF_DIR}")

    university_folders = [p for p in PDF_DIR.iterdir() if p.is_dir()]
    if not university_folders:
        raise ValueError(
            f"No university folders found in {PDF_DIR}. "
            "Create one folder per university, e.g. data/pdfs/NUST/"
        )

    all_docs = []
    for uni_folder in university_folders:
        university_name = uni_folder.name
        pdf_files = list(uni_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"  [!] No PDFs found for {university_name}, skipping.")
            continue

        for pdf_path in pdf_files:
            print(f"  Loading {university_name}/{pdf_path.name} ...")
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            category = guess_category(pdf_path.name)

            for page in pages:
                page.metadata["university"] = university_name
                page.metadata["category"] = category
                page.metadata["source_file"] = pdf_path.name
            all_docs.extend(pages)

    return all_docs


def build_vectorstore():
    print(f"Scanning {PDF_DIR} for university PDFs...\n")
    raw_docs = load_and_tag_documents()
    print(f"\nLoaded {len(raw_docs)} pages total.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks.")

    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
    )

    store = SimpleVectorStore(VECTOR_STORE_PATH)
    store.clear()  # rebuild from scratch each time ingest runs

    print(f"\nEmbedding {len(chunks)} chunks (this calls the Gemini API, "
          f"may take a few minutes on the free tier)...")

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Free tier allows ~100 embedding requests/minute. Embed one chunk at a
    # time with a small delay so we never burst past that, and automatically
    # wait and retry if a 429 (rate limit) error slips through anyway.
    all_embeddings = []
    for i, text in enumerate(texts):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                vector = embedder.embed_query(text)
                all_embeddings.append(vector)
                break
            except ResourceExhausted:
                wait = 25 * (attempt + 1)
                print(f"  Rate limit hit, waiting {wait}s before retrying "
                      f"chunk {i + 1}/{len(texts)}...")
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"Failed to embed chunk {i + 1} after {max_retries} retries. "
                "Check your Gemini API quota at https://ai.dev/rate-limit"
            )

        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            print(f"  Embedded {i + 1}/{len(texts)} chunks")

        if (i + 1) % 20 == 0:
            # Save progress periodically so a crash doesn't lose everything
            store.records = []
            store.add(
                texts=texts[:len(all_embeddings)],
                metadatas=metadatas[:len(all_embeddings)],
                embeddings=all_embeddings,
            )
            store.save()

        time.sleep(0.7)  # stay comfortably under 100 requests/minute

    store.records = []
    store.add(texts=texts, metadatas=metadatas, embeddings=all_embeddings)
    store.save()

    print(f"\nDone. Saved {len(store.records)} chunks to {VECTOR_STORE_PATH}")
    return store


if __name__ == "__main__":
    build_vectorstore()
