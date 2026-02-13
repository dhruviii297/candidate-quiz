CHROMA_BASE_URL = os.getenv("CHROMA_BASE_URL", "")  # e.g. https://your-chroma.onrender.com
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default_database")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "quizzes")

def chroma_base():
    if not CHROMA_BASE_URL:
        raise HTTPException(status_code=500, detail="CHROMA_BASE_URL not set")
    return CHROMA_BASE_URL.rstrip("/")

def chroma_get_or_create_collection_id() -> str:
    # List collections, find by name; if missing, create
    list_url = f"{chroma_base()}/api/v2/tenants/{CHROMA_TENANT}/databases/{CHROMA_DATABASE}/collections"
    r = requests.get(list_url, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Chroma list collections failed: {r.text}")
    cols = r.json() if isinstance(r.json(), list) else []
    found = next((c for c in cols if c.get("name") == CHROMA_COLLECTION), None)
    if found:
        return found["id"]

    create_url = f"{chroma_base()}/api/v2/tenants/{CHROMA_TENANT}/databases/{CHROMA_DATABASE}/collections"
    payload = {"name": CHROMA_COLLECTION, "get_or_create": True, "metadata": {"purpose": "quiz-agent-poc"}}
    r2 = requests.post(create_url, json=payload, timeout=30)
    if r2.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Chroma create collection failed: {r2.text}")
    return r2.json()["id"]

def chroma_query_similar(collection_id: str, query_text: str, n_results: int = 3):
    # NOTE: Chroma query endpoint usually expects query_embeddings.
    # For a PoC we store/query by document text ONLY if your Chroma server supports server-side embedding.
    # If your server does NOT, you’ll need embeddings (I can give that patch).
    url = f"{chroma_base()}/api/v2/tenants/{CHROMA_TENANT}/databases/{CHROMA_DATABASE}/collections/{collection_id}/query"
    payload = {"query_texts": [query_text], "n_results": n_results, "include": ["documents", "metadatas", "distances"]}
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Chroma query failed: {r.text}")
    data = r.json()
    return (data.get("documents") or [[]])[0] or []

def chroma_upsert_quiz(collection_id: str, quiz_id: str, quiz_doc: str, metadata: dict):
    url = f"{chroma_base()}/api/v2/tenants/{CHROMA_TENANT}/databases/{CHROMA_DATABASE}/collections/{collection_id}/upsert"
    payload = {"ids": [quiz_id], "documents": [quiz_doc], "metadatas": [metadata]}
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Chroma upsert failed: {r.text}")
