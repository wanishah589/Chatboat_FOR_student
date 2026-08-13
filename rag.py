from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from app.config import (
    VECTOR_STORE_PATH, GOOGLE_API_KEY, EMBEDDING_MODEL, LLM_MODEL, RETRIEVAL_K,
)
from app.vectorstore import SimpleVectorStore

SYSTEM_PROMPT = """You are an admissions guidance assistant for Pakistani university applicants.

Your job has three parts:
1. FIELD FIT: If the student hasn't shared their intermediate group (pre-medical /
   pre-engineering / ICS / commerce), their approximate aggregate %, and what subjects
   or activities interest them, ask for these first in one short, friendly message
   before recommending a field. Do not ask more than 3 questions total.
2. UNIVERSITY MATCH: Once you know their field and aggregate, use ONLY the retrieved
   context below to tell them which universities they are likely eligible for, and
   what the actual eligibility criteria is. If the context doesn't cover a university
   they ask about, say you don't have that data yet rather than guessing.
3. TEST INFO: When asked what entrance test a university requires, answer using the
   retrieved context's test_type information for that specific university only.

Rules:
- Never invent deadlines, percentages, or test names that aren't in the context.
- Always mention which university a fact is about, since students may ask about
  several in one conversation.
- Keep answers short and conversational, suitable for a chat widget, not an essay.
- If context is empty or irrelevant to the question, say so and suggest checking the
  official university website.
- You may respond in English or Roman Urdu depending on how the student writes to you.

Retrieved context:
{context}
"""


def format_context(records: list[dict]) -> str:
    if not records:
        return "(no relevant data found)"
    parts = []
    for r in records:
        uni = r["metadata"].get("university", "Unknown")
        cat = r["metadata"].get("category", "general")
        parts.append(f"[{uni} | {cat}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


class AdmissionsRAG:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.embedder = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
        )
        self.store = SimpleVectorStore(VECTOR_STORE_PATH)
        if not self.store.records:
            print(
                "[warning] Vector store is empty. Run 'python -m app.ingest' "
                "after adding PDFs to data/pdfs/<University>/"
            )
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.3
        )

    def retrieve(self, query: str, university: str | None = None):
        query_embedding = self.embedder.embed_query(query)
        filter_metadata = {"university": university} if university else None
        return self.store.similarity_search(
            query_embedding, k=RETRIEVAL_K, filter_metadata=filter_metadata
        )

    def answer(self, message: str, history: list[dict] | None = None,
               university: str | None = None) -> str:
        history = history or []
        records = self.retrieve(message, university=university)
        context = format_context(records)

        messages = [("system", SYSTEM_PROMPT.format(context=context))]
        for turn in history[-6:]:  # keep last 6 turns to stay light on tokens
            role = "human" if turn["role"] == "user" else "ai"
            messages.append((role, turn["content"]))
        messages.append(("human", message))

        response = self.llm.invoke(messages)
        return response.content


# Loaded once and reused across requests (see app/main.py)
_rag_instance: AdmissionsRAG | None = None


def get_rag() -> AdmissionsRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = AdmissionsRAG()
    return _rag_instance
