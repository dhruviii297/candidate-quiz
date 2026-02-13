import os
import json
import time
import requests
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import chromadb


# ---------- Config ----------
MEM0_BASE_URL = os.getenv("MEM0_BASE_URL", "https://api.mem0.ai")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

CHROMA_HOST = os.getenv("CHROMA_HOST", "")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "quizzes")

# Shared secret so n8n (or your portal) can call this safely
AGENT_SECRET = os.getenv("AGENT_SECRET", "")


# ---------- FastAPI ----------
app = FastAPI(title="Quiz Agent API (Mem0 + Chroma + OpenRouter)")


class QuizGenerateRequest(BaseModel):
    candidate_id: str = Field(..., description="Unique candidate identifier")
    role: str
    skills: List[str] = []
    difficulty: str = "medium"
    num_questions: int = 8


def require_secret(x_agent_secret: Optional[str]):
    if not AGENT_SECRET:
        # allow no-secret mode for very quick local PoC
        return
    if x_agent_secret != AGENT_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def mem0_headers() -> Dict[str, str]:
    if not MEM0_API_KEY:
        raise HTTPException(status_code=500, detail="MEM0_API_KEY not set")
    return {
        "Authorization": f"Token {MEM0_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def openrouter_headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def get_chroma_collection():
    if not CHROMA_HOST:
        raise HTTPException(status_code=500, detail="CHROMA_HOST not set")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


def mem0_search_candidate(candidate_id: str) -> List[str]:
    # Mem0 search endpoint (v2) lets you filter by user_id
    # (If your workspace uses a different variant, adjust accordingly.)
    url = f"{MEM0_BASE_URL}/v2/memories/search"
    payload = {
        "query": "candidate quiz preferences and history",
        "filters": {"AND": [{"user_id": candidate_id}]},
        "top_k": 5,
        "version": "v2",
    }
    r = requests.post(url, headers=mem0_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Mem0 search failed: {r.text}")
    data = r.json()
    # mem0 returns list-like items; normalize to strings
    memories = []
    if isinstance(data, list):
        for item in data:
            m = item.get("memory") or item.get("text") or ""
            if m:
                memories.append(m)
    return memories


def mem0_add_summary(candidate_id: str, summary: Dict[str, Any]):
    # Mem0 add endpoint: POST /v1/memories/ with Authorization: Token <key> :contentReference[oaicite:1]{index=1}
    url = f"{MEM0_BASE_URL}/v1/memories/"
    payload = {
        "user_id": candidate_id,
        "messages": [
            {"role": "user", "content": "Quiz generated for recruitment assessment."},
            {"role": "assistant", "content": json.dumps(summary)},
        ],
        "metadata": {"source": "quiz-agent-poc", "kind": "quiz_summary"},
    }
    r = requests.post(url, headers=mem0_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Mem0 add failed: {r.text}")


def chroma_similar_quizzes(collection, query_text: str, n_results: int = 3) -> List[str]:
    # For PoC: let Chroma embed with its default embedding function? Not ideal.
    # Better PoC: store raw text and query via "query_texts" (Chroma will embed if configured).
    # If your Chroma server has no embedding function, you can instead store and query using your own embeddings.
    try:
        res = collection.query(query_texts=[query_text], n_results=n_results)
        docs = res.get("documents", [[]])[0] or []
        return [d for d in docs if d]
    except Exception:
        return []


def chroma_store_quiz(collection, candidate_id: str, quiz: Dict[str, Any]):
    doc = json.dumps(quiz)
    quiz_id = f"quiz_{candidate_id}_{int(time.time()*1000)}"
    collection.upsert(
        ids=[quiz_id],
        documents=[doc],
        metadatas=[{
            "candidate_id": candidate_id,
            "role": quiz.get("role"),
            "difficulty": quiz.get("difficulty"),
            "quiz_title": quiz.get("quiz_title"),
            "source": "quiz-agent-poc"
        }]
    )


def openrouter_generate_quiz(req: QuizGenerateRequest, memories: List[str], similar_quizzes: List[str]) -> Dict[str, Any]:
    # OpenRouter is OpenAI-compatible; chat completions endpoint is documented. :contentReference[oaicite:2]{index=2}
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    prompt = f"""
Create a recruitment quiz as STRICT JSON only.

Schema:
{{
  "quiz_title": string,
  "role": string,
  "difficulty": "easy"|"medium"|"hard",
  "questions": [
    {{
      "id": string,
      "type": "mcq"|"short",
      "question": string,
      "options": [string] (only for mcq),
      "answer": string,
      "rubric": string
    }}
  ]
}}

REQUEST:
Role: {req.role}
Skills: {", ".join(req.skills)}
Difficulty: {req.difficulty}
Number of questions: {req.num_questions}

CANDIDATE MEMORY (use to adapt difficulty/style; do not include personal data):
- {" | ".join(memories) if memories else "none"}

SIMILAR PAST QUIZZES (use for style inspiration; do not copy verbatim):
- {" | ".join(similar_quizzes) if similar_quizzes else "none"}

Return JSON only.
"""
    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": "You generate recruitment quizzes. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(url, headers=openrouter_headers(), json=payload, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"OpenRouter failed: {r.text}")
    content = r.json()["choices"][0]["message"]["content"]
    try:
        quiz = json.loads(content)
    except Exception:
        raise HTTPException(status_code=502, detail=f"Model did not return valid JSON: {content[:500]}")
    return quiz


@app.post("/quiz/generate")
def generate_quiz(payload: QuizGenerateRequest, x_agent_secret: Optional[str] = Header(default=None)):
    require_secret(x_agent_secret)

    memories = mem0_search_candidate(payload.candidate_id)
    collection = get_chroma_collection()

    query_text = f"Role={payload.role}; Skills={','.join(payload.skills)}; Diff={payload.difficulty}"
    similar = chroma_similar_quizzes(collection, query_text=query_text, n_results=3)

    quiz = openrouter_generate_quiz(payload, memories, similar)

    # Store summary to Mem0 (short + meaningful)
    summary = {
        "type": "quiz_generated",
        "quiz_title": quiz.get("quiz_title"),
        "role": quiz.get("role", payload.role),
        "difficulty": quiz.get("difficulty", payload.difficulty),
        "num_questions": len(quiz.get("questions", []) or []),
        "skills_hint": payload.skills[:10],
    }
    mem0_add_summary(payload.candidate_id, summary)

    # Store full quiz to Chroma for similarity reuse
    chroma_store_quiz(collection, payload.candidate_id, quiz)

    return quiz


@app.get("/health")
def health():
    return {"ok": True}
