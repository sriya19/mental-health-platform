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
import json
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
    Enhanced robust search using AI to understand intent and find accurate, relevant datasets.
    Uses multi-query strategy and relevance filtering for best results.
    """
    results = []

    # Step 1: Use AI to deeply understand the search intent and extract structured information
    try:
        intent_prompt = f"""Analyze this health data search query and extract structured information.

Query: "{q.story}"

Extract and return in this exact JSON format:
{{
  "medical_terms": ["primary medical condition/topic", "related terms/synonyms"],
  "location": "geographic location if mentioned, or null",
  "demographics": "population group if mentioned (age, gender, race, etc.), or null",
  "time_period": "time period if mentioned (year, date range), or null",
  "data_type": "type of data needed (surveillance, survey, rates, statistics, etc.)",
  "search_queries": ["best search query", "alternative query 1", "alternative query 2"]
}}

Focus on medical/public health terminology. Include synonyms and related terms.
For search_queries, create 3 variations that would find relevant CDC/SAMHSA datasets."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=_headers_common,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": intent_prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"}
                },
                timeout=15.0
            )
            response.raise_for_status()
            intent_data = response.json()["choices"][0]["message"]["content"]
            search_intent = json.loads(intent_data)

            print(f"[Semantic Search] AI extracted intent: {search_intent}")

            # Get search query variations
            search_queries = search_intent.get("search_queries", [q.story])
            medical_terms = search_intent.get("medical_terms", [])
            location = search_intent.get("location")

    except Exception as e:
        print(f"[Semantic Search] Intent extraction failed: {e}, using fallback")
        # Fallback to simple query
        search_queries = [q.story]
        medical_terms = q.story.lower().split()[:5]
        location = None

    print(f"[Semantic Search] Search queries: {search_queries}")

    # Step 2: Determine which organizations to search
    orgs_to_search = []
    include_local = False

    if q.org == "All":
        # Search all available organizations
        orgs_to_search = ["CDC", "SAMHSA"]
        # Note: BRFSS, NHIS, YRBSS, NSSP are all CDC surveys, so CDC search covers them
        include_local = True  # Also search local Baltimore datasets
    elif q.org == "LOCAL":
        include_local = True
        orgs_to_search = []  # Only search local datasets
    elif q.org in ["CDC", "SAMHSA", "BRFSS", "NHIS", "YRBSS", "NSSP"]:
        orgs_to_search = [q.org]
    else:
        # Default to CDC if unknown
        orgs_to_search = ["CDC"]

    # Step 3: Multi-query search strategy - search with multiple query variations
    seen_uids = set()  # Avoid duplicates

    for search_query in search_queries[:3]:  # Use top 3 query variations
        print(f"[Semantic Search] Searching with query: '{search_query}'")

        for org in orgs_to_search:
            try:
                # Search with current query variation
                online_hits = await socrata.search_catalog(
                    org,
                    search_query,
                    limit=max(10, q.k // (len(search_queries) * len(orgs_to_search)))
                )

                for hit in online_hits:
                    uid = hit.get("assetId") or hit.get("uid") or ""

                    # Skip if already found or invalid
                    if not uid or uid in seen_uids:
                        continue

                    seen_uids.add(uid)
                    results.append({
                        "uid": uid,
                        "name": hit.get("name", ""),
                        "description": hit.get("description", ""),
                        "org": org,
                        "source": "socrata",
                        "matched_query": search_query  # Track which query found this
                    })

                print(f"[Semantic Search] Query '{search_query[:30]}...' found {len(online_hits)} from {org}")
            except Exception as e:
                print(f"[Semantic Search] Query failed for {org}: {e}")

    # Step 4: Search locally uploaded datasets (keyword search on name/description)
    with engine.begin() as conn:
        if medical_terms and len(medical_terms) > 0:
            search_terms = medical_terms
        else:
            search_terms = [q.story]

        if include_local:
            # Build OR conditions for each keyword
            keyword_conditions = " OR ".join([f"name ILIKE :kw{i} OR description ILIKE :kw{i}" for i in range(len(search_terms))])

            local_sql = text(f"""
                SELECT
                    source_url,
                    name,
                    description,
                    source_org as org,
                    'local' as source
                FROM datasets
                WHERE ({keyword_conditions})
                ORDER BY first_ingested_at DESC
                LIMIT :k
            """)

            # Build parameters dict
            params = {"k": q.k // 2}
            for i, kw in enumerate(search_terms):
                params[f"kw{i}"] = f"%{kw}%"

            local_rows = conn.execute(
                local_sql,
                params
            ).mappings().all()

            print(f"[Semantic Search] Found {len(local_rows)} local datasets")
        else:
            # Build OR conditions for each keyword
            keyword_conditions = " OR ".join([f"name ILIKE :kw{i} OR description ILIKE :kw{i}" for i in range(len(search_terms))])

            local_sql = text(f"""
                SELECT
                    source_url,
                    name,
                    description,
                    source_org as org,
                    'local' as source
                FROM datasets
                WHERE source_org = :o
                  AND ({keyword_conditions})
                ORDER BY first_ingested_at DESC
                LIMIT :k
            """)

            # Build parameters dict
            params = {"o": q.org, "k": q.k // 2}
            for i, kw in enumerate(search_terms):
                params[f"kw{i}"] = f"%{kw}%"

            local_rows = conn.execute(
                local_sql,
                params
            ).mappings().all()

        # Convert to dicts and extract UIDs
        for row in local_rows:
            row_dict = dict(row)
            source_url = row_dict.get('source_url', '')

            # Extract UID from source_url (handles "local/abc123" format)
            if '/' in source_url:
                uid = source_url.split('/')[-1]
            else:
                uid = source_url

            # Skip duplicates
            if uid not in seen_uids:
                seen_uids.add(uid)
                row_dict['uid'] = uid
                results.append(row_dict)

    print(f"[Semantic Search] Total results before filtering: {len(results)}")

    # Step 5: AI-powered relevance filtering and ranking
    if len(results) > q.k:
        try:
            # Create dataset summaries for AI to evaluate
            dataset_summaries = []
            for i, r in enumerate(results[:min(50, len(results))]):  # Limit to first 50 for performance
                dataset_summaries.append({
                    "index": i,
                    "name": r.get("name", "")[:100],
                    "description": r.get("description", "")[:200]
                })

            relevance_prompt = f"""Given this user query: "{q.story}"

Rank these datasets by relevance (most relevant first). Return ONLY a JSON array of indices in order of relevance.
Consider medical terminology, topic match, and data usefulness.

Datasets:
{json.dumps(dataset_summaries, indent=2)}

Return format: {{"ranked_indices": [3, 1, 7, 2, ...]}}"""

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OPENAI_BASE_URL}/chat/completions",
                    headers=_headers_common,
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": relevance_prompt}],
                        "temperature": 0.2,
                        "max_tokens": 500,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                ranking_data = response.json()["choices"][0]["message"]["content"]
                ranking = json.loads(ranking_data)

                ranked_indices = ranking.get("ranked_indices", list(range(len(results))))
                print(f"[Semantic Search] AI ranked top 5 indices: {ranked_indices[:5]}")

                # Reorder results based on AI ranking
                reordered_results = []
                for idx in ranked_indices:
                    if idx < len(results):
                        reordered_results.append(results[idx])

                # Add any remaining results not in ranking
                for i, r in enumerate(results):
                    if i not in ranked_indices:
                        reordered_results.append(r)

                results = reordered_results

        except Exception as e:
            print(f"[Semantic Search] AI ranking failed: {e}, using original order")

    return {
        "used_semantic": True,
        "results": results[:q.k],  # Limit total results
        "model": OPENAI_EMBEDDING_MODEL,
        "total_found": len(results),
        "sources": {
            "socrata": len([r for r in results[:q.k] if r.get('source') == 'socrata']),
            "local": len([r for r in results[:q.k] if r.get('source') == 'local'])
        }
    }
