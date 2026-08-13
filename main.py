from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import ALLOWED_ORIGIN
from app.rag import get_rag

app = FastAPI(title="University Admissions Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []
    university: str | None = None  # optional filter, e.g. "NUST"


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        rag = get_rag()
        reply = rag.answer(
            message=req.message,
            history=[turn.model_dump() for turn in req.history],
            university=req.university,
        )
        return ChatResponse(reply=reply)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")
