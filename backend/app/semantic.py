# backend/app/semantic.py

from __future__ import annotations
from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import text
from .db import engine
from . import socrata

from sqlalchemy import bindparam

def _to_pgvector_literal(vec: list[float]) -> str:
    """
    Convert a Python list of floats to pgvector's text format: "[v1,v2,...]".
    Keep reasonable precision to avoid giant strings.
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


# ---- std / third-party ----
import time
import os
import asyncio
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

router = APIRouter(prefix="/semantic", tags=["semantic"])

# =========================
#   Configuration (env)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Determine embedding dimension based on model
def _dims_for_model(model: str) -> int:
    # OpenAI current dimensions: small=1536, large=3072
    return 3072 if "large" in model else 1536

EMBED_DIM = _dims_for_model(OPENAI_EMBEDDING_MODEL)

# Throttle (requests per second)
EMBED_RPS = float(os.getenv("EMBED_RPS", "1"))
EMBEDDINGS_ENDPOINT = f"{OPENAI_BASE_URL}/embeddings"

_headers_common = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

# =========================
#   Retry + Embedding helper
# =========================
def _should_retry(exc: Exception) -> bool:
    """Retry on OpenAI 429/5xx HTTP status errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
async def embed_text(text: str):
    """
    Create an embedding and log estimated cost + latency for transparency.
    """
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": OPENAI_EMBEDDING_MODEL, "input": text}

    start = time.time()
    async with httpx.AsyncClient() as client:
        r = await client.post(EMBEDDINGS_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    duration = round(time.time() - start, 2)
    usage = data.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    cost_estimate = (total_tokens / 1_000_000) * 0.02  # ~ $0.02 per 1M tokens

    print(
        f"[COST LOG] Embedded {len(text)} chars in {duration}s | "
        f"Tokens: {total_tokens} | Estimated Cost: ${cost_estimate:.6f}"
    )

    return data["data"][0]["embedding"]

# =========================
#   Pydantic models
# =========================
class SemanticQuery(BaseModel):
    story: str
    org: str
    k: int = 10
    persona: str | None = None

# =========================
#   Endpoints
# =========================
@router.post("/reindex")
async def reindex(
    org: str = Query(..., pattern="^(CDC|SAMHSA)$"),
    limit: int = Query(50, ge=1, le=500),
    q: str | None = Query(None, description="Optional catalog query to filter datasets"),
):
    """
    Rebuild or refresh semantic_index for the given org from Socrata catalog hits.
    """
    catalog_query = q or ""
    hits = await socrata.search_catalog(org, catalog_query, limit=limit)
    if not hits:
        return {"ok": True, "indexed": 0, "skipped": 0, "reason": "no_hits"}

    delay = 1.0 / max(EMBED_RPS, 0.1)
    indexed = 0
    skipped = 0

    with engine.begin() as conn:
        # Create table if missing
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS semantic_index (
                    id BIGSERIAL PRIMARY KEY,
                    org TEXT NOT NULL,
                    uid TEXT NOT NULL,
                    name TEXT,
                    description TEXT,
                    embedding vector({EMBED_DIM}),
                    created_at TIMESTAMPTZ DEFAULT now(),
                    UNIQUE(org, uid)
                );
                CREATE INDEX IF NOT EXISTS semantic_index_org_uid_idx ON semantic_index(org, uid);
                """
            )
        )

        for h in hits:
            uid = h.get("assetId") or h.get("uid") or ""
            if not uid:
                skipped += 1
                continue

            name = (h.get("name") or "").strip()
            desc = (h.get("description") or "").strip()
            text_raw = (name + "\n\n" + desc).strip()[:8000] or name

            try:
                emb = await embed_text(text_raw)
                conn.execute(
                    text(
                        """
                        INSERT INTO semantic_index (org, uid, name, description, embedding)
                        VALUES (:o, :u, :n, :d, :e)
                        ON CONFLICT (org, uid) DO UPDATE
                        SET name = EXCLUDED.name,
                            description = EXCLUDED.description,
                            embedding = EXCLUDED.embedding
                        """
                    ),
                    {"o": org, "u": uid, "n": name, "d": desc, "e": emb},
                )
                indexed += 1
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as ex:
                if ex.response.status_code == 429:
                    return {
                        "ok": False,
                        "indexed": indexed,
                        "skipped": skipped,
                        "error": "rate_limited",
                        "hint": "Lower ?limit= and/or set EMBED_RPS=0.5 in .env",
                    }
                skipped += 1
            except Exception:
                skipped += 1

    return {
        "ok": True,
        "indexed": indexed,
        "skipped": skipped,
        "used_model": OPENAI_EMBEDDING_MODEL,
        "throttle_rps": EMBED_RPS,
    }

@router.post("/search")
async def semantic_search(q: SemanticQuery):
    """
    Semantic search over previously reindexed datasets (by org).
    If embeddings are currently rate-limited, falls back to keyword search.
    """
    # Try to embed the user story
    try:
        query_text = f"{q.persona}: {q.story}" if q.persona else q.story
        emb = await embed_text(query_text)
    except httpx.HTTPStatusError as ex:
        if ex.response.status_code == 429:
            hits = await socrata.search_catalog(q.org, q.story, limit=q.k)
            return {
                "used_semantic": False,
                "fallback": "keyword",
                "reason": "rate_limited",
                "results": hits[: q.k],
            }
        raise
    # Format embedding for pgvector and CAST it server-side
    vec_literal = _to_pgvector_literal(emb)

    sql = text("""
        SELECT uid, name, description
        FROM semantic_index
        WHERE org = :o
        ORDER BY embedding <-> CAST(:v AS vector)
        LIMIT :k
    """).bindparams(bindparam("v"))

    with engine.begin() as conn:
        rows = conn.execute(
            sql,
            {"o": q.org, "v": vec_literal, "k": q.k},
        ).mappings().all()


    return {
        "used_semantic": True,
        "results": [dict(r) for r in rows],
        "model": OPENAI_EMBEDDING_MODEL,
    }
