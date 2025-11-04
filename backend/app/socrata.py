import re
import os
from typing import Dict, Any, List, Optional

import httpx
from .config import settings

# Global Socrata Catalog v1 (cross-domain search)
CATALOG_V1 = "https://api.us.socrata.com/api/catalog/v1"

# Domains we support
DOMAINS = {
    "CDC": "data.cdc.gov",
    "SAMHSA": "data.samhsa.gov",
}

# Regex to extract UID from permalinks like /d/xkb8-kh2a or /dataset/xkb8-kh2a
UID_RE = re.compile(r"/(?:d|dataset)/([a-z0-9]{4}-[a-z0-9]{4})(?:[/?#]|$)", re.I)

# LLM Configuration for query extraction
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _headers() -> Dict[str, str]:
    h = {"accept": "application/json"}
    if settings.SOCRATA_APP_TOKEN:
        h["X-App-Token"] = settings.SOCRATA_APP_TOKEN
    return h


async def _llm_extract_search_terms(query: str) -> str:
    """
    Use LLM to intelligently extract optimal search keywords from verbose user stories.
    This is much more reliable than regex-based extraction.
    
    Falls back to original query if LLM fails or API key is missing.
    """
    
    # If already short and concise, don't need LLM
    if len(query) <= 50:
        return query
    
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        print("[LLM Query Extraction] No OpenAI API key found, using original query")
        return query
    
    prompt = f"""You are a search query optimizer for CDC and SAMHSA health datasets. 
Your job is to extract the most relevant search keywords from verbose user stories.

Rules:
1. Extract 3-8 keywords that would find relevant health datasets
2. Keep domain-specific terms: mental health, substance abuse, overdose, suicide, depression, anxiety, etc.
3. Keep important qualifiers: 
   - Demographics (age groups, gender, race/ethnicity)
   - Geography (state names, county, city, region)
   - Time periods (years, date ranges)
4. Remove user story fluff: "As a...", "I want to...", "so that...", "I need to..."
5. Use proper terminology: "overdose" not "OD", "suicide" not "self-harm"
6. Keep survey names if mentioned: BRFSS, YRBSS, NVSS, etc.

Examples:
Input: "As a public health researcher, I want to analyze drug overdose death rates by state and county so that I can identify high-risk areas for intervention."
Output: drug overdose death rates state county

Input: "I need to investigate the relationship between mental distress and chronic conditions like diabetes and heart disease using population survey data."
Output: mental distress chronic conditions diabetes heart disease population survey

Input: "As a data analyst, I'm looking for datasets on mental health in Virginia in 2023-2024"
Output: mental health Virginia 2023 2024

Input: "As an epidemiologist, I want to examine suicide rates among veterans over the past decade."
Output: suicide rates veterans decade trends

Input: "I'm studying mental health risk behaviors among high school students including depression and substance use."
Output: mental health risk behaviors high school students depression substance use

Now extract keywords from this query:
"{query}"

Respond with ONLY the extracted keywords (no explanation, no preamble, no extra text):"""

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",  # Fast and cheap ($0.000150/1K input tokens)
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a search query optimizer. Extract only relevant keywords, nothing else."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,  # Low temperature for consistency
                    "max_tokens": 100     # Short response
                }
            )
            response.raise_for_status()
            data = response.json()
            extracted = data["choices"][0]["message"]["content"].strip()
            
            # Remove any quotes or extra formatting
            extracted = extracted.strip('"').strip("'").strip()
            
            print(f"[LLM Query Extraction] Original: '{query[:80]}...'")
            print(f"[LLM Query Extraction] Extracted: '{extracted}'")
            print(f"[LLM Query Extraction] Cost: ~$0.0001")
            
            return extracted
            
    except httpx.HTTPStatusError as e:
        print(f"[LLM Query Extraction] HTTP Error {e.response.status_code}: {e.response.text[:100]}")
        return query  # Fallback to original
        
    except Exception as e:
        print(f"[LLM Query Extraction] Error: {e}. Falling back to original query.")
        return query  # Fallback to original


async def search_catalog(org: str, q: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search Socrata catalog with LLM-powered query preprocessing.
    
    Strategy:
    1) Use LLM to extract search terms from verbose queries
    2) Try global catalog v1 (api.us.socrata.com)
    3) If it 404s or errors, fall back to site-local search API
    4) Normalize results to: name, description, uid, link, org
    
    Args:
        org: Organization ("CDC" or "SAMHSA")
        q: User's search query (can be verbose)
        limit: Maximum number of results to return
        
    Returns:
        List of dataset dictionaries with name, description, uid, link, org
    """
    # Use LLM to extract meaningful search terms from verbose queries
    search_terms = await _llm_extract_search_terms(q)
    
    domain = DOMAINS[org]

    async with httpx.AsyncClient(timeout=30) as client:
        # --- Attempt 1: global catalog v1 with LLM-extracted terms
        params = {"q": search_terms, "domains": domain, "only": "datasets", "limit": limit}
        try:
            r = await client.get(CATALOG_V1, headers=_headers(), params=params)
            r.raise_for_status()
            data = r.json()
            out: List[Dict[str, Any]] = []
            for item in data.get("results", []):
                res = item.get("resource") or {}
                name = res.get("name")
                desc = (res.get("description") or "").strip()
                permalink = item.get("permalink") or ""
                uid = res.get("id")
                # Fallback: parse UID from permalink if missing
                if not uid and permalink:
                    m = UID_RE.search(permalink)
                    if m:
                        uid = m.group(1)
                out.append({
                    "name": name, 
                    "description": desc, 
                    "uid": uid,
                    "link": permalink, 
                    "org": org
                })
            if out:
                print(f"[Catalog Search] Found {len(out)} results via global catalog")
                return out
        except httpx.HTTPStatusError as e:
            # Some tenants (e.g., SAMHSA) occasionally 404 on the global catalog
            print(f"[Catalog Search] Global catalog failed ({e.response.status_code}), trying local...")
            if e.response.status_code not in (404, 400, 429, 500):
                raise

        # --- Attempt 2: site-local search (Socrata's legacy search api)
        # Example: https://data.samhsa.gov/api/search/views.json?q=treatment&limit=10
        local_url = f"https://{domain}/api/search/views.json"
        params2 = {"q": search_terms, "limit": limit}
        
        try:
            r2 = await client.get(local_url, headers=_headers(), params=params2)
            r2.raise_for_status()
            data2 = r2.json()

            out2: List[Dict[str, Any]] = []
            for item in data2.get("results", []):
                # These objects use different keys; normalize:
                # Prefer 'id' (UID). Some responses nest in 'view' or 'resource'.
                uid = item.get("id") or (item.get("view") or {}).get("id") or (item.get("resource") or {}).get("id")
                name = item.get("name") or (item.get("view") or {}).get("name")
                desc = (item.get("description") or (item.get("view") or {}).get("description") or "").strip()

                # Build a permalink if missing
                link = item.get("permalink") or (item.get("view") or {}).get("permalink") \
                       or (f"https://{domain}/d/{uid}" if uid else "")

                # Final UID safeguard: parse from link if necessary
                if (not uid) and link:
                    m = UID_RE.search(link)
                    if m:
                        uid = m.group(1)

                out2.append({
                    "name": name, 
                    "description": desc, 
                    "uid": uid,
                    "link": link, 
                    "org": org
                })
            
            print(f"[Catalog Search] Found {len(out2)} results via local search")
            return out2
            
        except Exception as e:
            print(f"[Catalog Search] Local search also failed: {e}")
            return []  # Return empty list if both methods fail


async def fetch_rows(
    org: str, dataset_uid: str, limit: int = 5000, where: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Pull rows from the specific domain's resource endpoint.
    
    Args:
        org: Organization ("CDC" or "SAMHSA")
        dataset_uid: The 4x4 dataset identifier (e.g., "abc1-def2")
        limit: Maximum number of rows to fetch
        where: Optional SoQL WHERE clause for filtering
        
    Returns:
        List of row dictionaries
    """
    base = f"https://{DOMAINS[org]}"
    path = f"/resource/{dataset_uid}.json"
    params: Dict[str, Any] = {"$limit": limit}
    if where:
        params["$where"] = where

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(base + path, headers=_headers(), params=params)
        r.raise_for_status()
        return r.json()