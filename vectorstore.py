"""
A minimal, dependency-light replacement for ChromaDB.

Why this exists: chromadb pulls in chroma-hnswlib, which needs a C++ compiler
to build on Windows. For a project this size (a few hundred to a few thousand
chunks from 10 PDFs), we don't need a specialized vector database — plain
numpy cosine similarity is fast enough and has zero compilation requirements.

Storage format: a single JSON file containing, for each chunk, its text,
metadata, and embedding vector. Loaded fully into memory at startup.
"""

import json
import numpy as np
from pathlib import Path


class SimpleVectorStore:
    def __init__(self, path: Path):
        self.path = path
        self.records = []  # list of {"text": ..., "metadata": {...}, "embedding": [...]}
        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f)

    def add(self, texts: list[str], metadatas: list[dict], embeddings: list[list[float]]):
        for text, meta, emb in zip(texts, metadatas, embeddings):
            self.records.append({"text": text, "metadata": meta, "embedding": emb})

    def clear(self):
        self.records = []

    def similarity_search(self, query_embedding: list[float], k: int = 5,
                           filter_metadata: dict | None = None):
        if not self.records:
            return []

        candidates = self.records
        if filter_metadata:
            candidates = [
                r for r in candidates
                if all(r["metadata"].get(key) == val for key, val in filter_metadata.items())
            ]
        if not candidates:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec) or 1e-10

        scored = []
        for r in candidates:
            vec = np.array(r["embedding"])
            vec_norm = np.linalg.norm(vec) or 1e-10
            similarity = float(np.dot(query_vec, vec) / (query_norm * vec_norm))
            scored.append((similarity, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]
