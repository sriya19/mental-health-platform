# backend/app/main.py
from __future__ import annotations

from typing import Optional
import json
import io
import pandas as pd
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from tenacity import retry, stop_after_attempt, wait_exponential

from .db import engine
from . import socrata, ingest, interpret
from .config import settings

# Import the enhanced RAG module
try:
    from . import rag
except ImportError:
    rag = None

# Import Baltimore-specific indexer
try:
    from . import baltimore_indexer
except ImportError:
    baltimore_indexer = None

# FastAPI app
app = FastAPI(title="MH Backend", version="1.0.0")


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return JSONResponse(
        content={"ok": True, "db": True, "bucket": settings.S3_BUCKET},
        media_type="application/json",
    )


# -----------------------------
# Catalog Search (Socrata)
# -----------------------------
@app.get("/catalog/search")
async def catalog_search(
    org: str = Query(..., pattern="^(All|CDC|SAMHSA)$"),
    q: str = Query(..., description="Catalog search query"),
):
    # Determine which organizations to search
    if org == "All":
        orgs_to_search = ["CDC", "SAMHSA"]
    else:
        orgs_to_search = [org]

    # Search each organization and merge results
    all_results = []
    for search_org in orgs_to_search:
        try:
            hits = await socrata.search_catalog(search_org, q, limit=10)
            for h in hits:
                all_results.append({
                    "name": h.get("name"),
                    "description": h.get("description"),
                    "uid": h.get("uid"),
                    "link": h.get("link"),
                    "org": search_org,  # Use the actual org, not "All"
                })
        except Exception as e:
            print(f"[Catalog Search] Failed to search {search_org}: {e}")

    return {"results": all_results}


# -----------------------------
# Reliable fetch helper
# -----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _pull(org: str, uid: str):
    return await socrata.fetch_rows(org, uid, limit=5000)


# -----------------------------
# Request Models
# -----------------------------
class IngestRequest(BaseModel):
    org: str
    query: Optional[str] = None
    pick_uid: Optional[str] = None
    auto_index: bool = False  # New: automatically index for RAG after ingesting

class IndexRequest(BaseModel):
    org: str
    uid: str
    limit_rows: int = 20000


# -----------------------------
# Enhanced Ingest with Auto-Indexing
# -----------------------------
@app.post("/ingest")
async def do_ingest(req: IngestRequest):
    """
    Ingest dataset and optionally index it for data-based RAG queries.
    """
    import pandas as pd
    org = (req.org or "").upper()
    try:
        # Decide UID
        if req.pick_uid:
            uid = req.pick_uid
            name, desc, link = f"{org}:{uid}", "", f"https://data.{org.lower()}.gov/d/{uid}"
        else:
            if not req.query:
                return {"ingested": False, "reason": "missing_query_and_pick_uid"}
            hits = await socrata.search_catalog(org, req.query, limit=1)
            if not hits:
                return {"ingested": False, "reason": "no_catalog_results"}
            hit = hits[0]
            uid = hit.get("uid")
            link = hit.get("link") or ""
            if not uid and "/d/" in link:
                try:
                    uid = link.split("/d/")[1].split("?")[0].split("/")[0]
                except Exception:
                    uid = None
            if not uid:
                return {"ingested": False, "reason": "no_uid_in_result"}
            name = hit.get("name") or f"{org}:{uid}"
            desc = hit.get("description") or ""

        # Pull rows
        rows = await socrata.fetch_rows(org, uid, limit=50000)
        if not rows:
            return {"ingested": False, "reason": "no_rows_returned", "dataset_uid": uid}

        df = pd.DataFrame(rows)

        # Register meta
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO datasets (name, source_org, source_url, description)
                VALUES (:n,:o,:u,:d)
                ON CONFLICT (source_org, source_url) DO UPDATE 
                SET name = EXCLUDED.name, description = EXCLUDED.description;
            """), {"n": name, "o": org, "u": link, "d": desc})

        # Write to S3
        from . import ingest as ingest_mod
        s3_key = f"raw/{org}/{uid}.parquet"
        ingest_mod.write_parquet_to_minio(df, s3_key)
        
        result = {
            "ingested": True, 
            "rows": int(len(df)), 
            "dataset_uid": uid, 
            "s3_key": s3_key,
            "name": name
        }
        
        # Auto-index if requested
        if req.auto_index and rag:
            print(f"[Ingest] Auto-indexing dataset {uid}")
            try:
                index_result = await rag.index_dataset_content(
                    org=org,
                    uid=uid,
                    limit_rows=min(20000, len(df))
                )
                result["indexed"] = index_result.get("success", False)
                result["chunks_created"] = index_result.get("chunks_created", 0)
                print(f"[Ingest] Indexed {result['chunks_created']} chunks")
            except Exception as e:
                result["indexed"] = False
                result["index_error"] = str(e)
                print(f"[Ingest] Index failed: {e}")
        
        return result
        
    except Exception as e:
        return {"ingested": False, "error": str(e)}


# -----------------------------
# Upload Local CSV File
# -----------------------------
@app.post("/upload_csv")
async def upload_csv(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    description: str = Form(""),
    org: str = Form("LOCAL"),
    auto_index: bool = Form(False)
):
    """
    Upload a local CSV file and ingest it into the platform.
    Optionally index it for RAG queries.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        # Generate a unique ID for this dataset
        import hashlib
        import datetime
        uid = hashlib.md5(f"{dataset_name}_{datetime.datetime.now()}".encode()).hexdigest()[:8]

        # Register in database
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO datasets (name, source_org, source_url, description)
                VALUES (:n, :o, :u, :d)
                ON CONFLICT (source_org, source_url) DO UPDATE
                SET name = EXCLUDED.name, description = EXCLUDED.description;
            """), {
                "n": dataset_name,
                "o": org,
                "u": f"local/{uid}",
                "d": description or f"Uploaded CSV: {file.filename}"
            })

        # Write to S3/MinIO
        from . import ingest as ingest_mod
        s3_key = f"raw/{org}/{uid}.parquet"
        ingest_mod.write_parquet_to_minio(df, s3_key)

        result = {
            "ingested": True,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dataset_uid": uid,
            "s3_key": s3_key,
            "name": dataset_name,
            "org": org
        }

        # Auto-index if requested
        if auto_index and rag:
            print(f"[Upload CSV] Auto-indexing dataset {uid}")
            try:
                index_result = await rag.index_dataset_content(
                    org=org,
                    uid=uid,
                    limit_rows=min(20000, len(df))
                )
                result["indexed"] = index_result.get("success", False)
                result["chunks_created"] = index_result.get("chunks_created", 0)
                print(f"[Upload CSV] Indexed {result['chunks_created']} chunks")
            except Exception as e:
                result["indexed"] = False
                result["index_error"] = str(e)
                print(f"[Upload CSV] Index failed: {e}")

        return result

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


# -----------------------------
# Index existing dataset for RAG
# -----------------------------
@app.post("/index_dataset")
async def index_dataset_for_rag(req: IndexRequest):
    """
    Index an already-ingested dataset for data-based RAG queries.
    This creates searchable chunks from the actual data.
    """
    if not rag:
        raise HTTPException(
            status_code=500, 
            detail="Enhanced RAG module not available. Check rag.py import."
        )
    
    try:
        result = await rag.index_dataset_content(
            org=req.org,
            uid=req.uid,
            limit_rows=req.limit_rows
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Index Baltimore-specific data only
# -----------------------------
@app.post("/index_baltimore")
async def index_baltimore_dataset(req: IndexRequest):
    """
    Index ONLY Baltimore/Maryland data from a dataset.
    Filters out all non-Baltimore data before creating chunks.
    """
    if not baltimore_indexer:
        raise HTTPException(
            status_code=500,
            detail="Baltimore indexer module not available."
        )

    try:
        result = await baltimore_indexer.index_baltimore_data(
            org=req.org,
            uid=req.uid
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Batch index all datasets for Baltimore data
# -----------------------------
@app.post("/batch_index_baltimore")
async def batch_index_baltimore(org: str = Query(..., pattern="^(CDC|SAMHSA)$")):
    """
    Re-index all datasets to include ONLY Baltimore/Maryland data.
    This replaces existing chunks with Baltimore-filtered chunks.
    """
    if not baltimore_indexer:
        raise HTTPException(
            status_code=500,
            detail="Baltimore indexer module not available."
        )

    # Get all ingested datasets
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT source_url
                FROM datasets
                WHERE source_org = :o
            """),
            {"o": org}
        ).fetchall()

    results = []
    for row in rows:
        source_url = row[0]
        if "/d/" in source_url:
            try:
                uid = source_url.split("/d/")[1].split("?")[0].split("/")[0]
                print(f"[Batch Baltimore Index] Processing {uid}")
                result = await baltimore_indexer.index_baltimore_data(
                    org=org,
                    uid=uid
                )
                results.append({
                    "uid": uid,
                    "success": result.get("success", False),
                    "baltimore_rows": result.get("baltimore_rows", 0),
                    "chunks": result.get("chunks_created", 0),
                    "reason": result.get("reason", "")
                })
            except Exception as e:
                results.append({
                    "uid": uid,
                    "success": False,
                    "error": str(e)
                })

    successful = sum(1 for r in results if r.get("success"))
    total_baltimore_rows = sum(r.get("baltimore_rows", 0) for r in results)
    total_chunks = sum(r.get("chunks", 0) for r in results)

    return {
        "org": org,
        "total_datasets": len(results),
        "successful": successful,
        "failed": len(results) - successful,
        "total_baltimore_rows": total_baltimore_rows,
        "total_chunks_created": total_chunks,
        "details": results
    }


# -----------------------------
# Batch index multiple datasets
# -----------------------------
@app.post("/batch_index")
async def batch_index(org: str = Query(..., pattern="^(CDC|SAMHSA)$")):
    """
    Index all ingested datasets for the given organization.
    This enables data-based RAG for all datasets.
    """
    if not rag:
        raise HTTPException(
            status_code=500,
            detail="Enhanced RAG module not available."
        )
    
    # Get all ingested datasets
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT source_url 
                FROM datasets 
                WHERE source_org = :o
            """),
            {"o": org}
        ).fetchall()
    
    results = []
    for row in rows:
        source_url = row[0]
        if "/d/" in source_url:
            try:
                uid = source_url.split("/d/")[1].split("?")[0].split("/")[0]
                result = await rag.index_dataset_content(
                    org=org,
                    uid=uid,
                    limit_rows=20000
                )
                results.append({
                    "uid": uid,
                    "success": result.get("success", False),
                    "chunks": result.get("chunks_created", 0)
                })
            except Exception as e:
                results.append({
                    "uid": uid,
                    "success": False,
                    "error": str(e)
                })
    
    successful = sum(1 for r in results if r.get("success"))
    return {
        "org": org,
        "total_datasets": len(results),
        "successful": successful,
        "failed": len(results) - successful,
        "details": results
    }


# -----------------------------
# Check indexing status
# -----------------------------
@app.get("/rag_status")
async def rag_status(org: str = Query("All")):
    """
    Check the status of data indexing for RAG.
    """
    with engine.begin() as conn:
        # Check if data_chunks table exists
        table_check = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'data_chunks'
                )
            """)
        ).scalar()

        if not table_check:
            return {
                "org": org,
                "status": "not_initialized",
                "indexed_datasets": 0,
                "total_chunks": 0,
                "message": "Data indexing not yet initialized. Ingest and index some datasets first."
            }

        # Get statistics - handle "All" organization
        if org == "All":
            stats = conn.execute(
                text("""
                    SELECT
                        COUNT(DISTINCT dataset_uid) as indexed_datasets,
                        COUNT(*) as total_chunks
                    FROM data_chunks
                """)
            ).fetchone()

            # Get list of indexed datasets with names
            datasets = conn.execute(
                text("""
                    SELECT
                        dc.dataset_uid,
                        dc.org,
                        COUNT(*) as chunk_count,
                        MAX(dc.created_at) as last_indexed,
                        d.name as dataset_name
                    FROM data_chunks dc
                    LEFT JOIN datasets d ON d.source_url LIKE '%' || dc.dataset_uid || '%'
                    GROUP BY dc.dataset_uid, dc.org, d.name
                    ORDER BY last_indexed DESC
                    LIMIT 10
                """)
            ).mappings().all()
        else:
            stats = conn.execute(
                text("""
                    SELECT
                        COUNT(DISTINCT dataset_uid) as indexed_datasets,
                        COUNT(*) as total_chunks
                    FROM data_chunks
                    WHERE org = :o
                """),
                {"o": org}
            ).fetchone()

            # Get list of indexed datasets with names
            datasets = conn.execute(
                text("""
                    SELECT
                        dc.dataset_uid,
                        COUNT(*) as chunk_count,
                        MAX(dc.created_at) as last_indexed,
                        d.name as dataset_name
                    FROM data_chunks dc
                    LEFT JOIN datasets d ON d.source_url LIKE '%' || dc.dataset_uid || '%'
                    WHERE dc.org = :o
                    GROUP BY dc.dataset_uid, d.name
                    ORDER BY last_indexed DESC
                    LIMIT 10
                """),
                {"o": org}
            ).mappings().all()
        
        return {
            "org": org,
            "status": "ready" if stats[0] > 0 else "empty",
            "indexed_datasets": stats[0],
            "total_chunks": stats[1],
            "recent_datasets": [dict(d) for d in datasets],
            "message": f"{stats[0]} datasets indexed with {stats[1]} searchable data chunks"
        }


# -----------------------------
# Datasets we have registered with index status
# -----------------------------
@app.get("/datasets")
def list_datasets(org: Optional[str] = None):
    """
    Return datasets with their indexing status
    """
    rows = []
    with engine.begin() as conn:
        # Check if data_chunks table exists
        has_chunks_table = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'data_chunks'
                )
            """)
        ).scalar()
        
        if org and org != "All":
            r = conn.execute(
                text("""
                    SELECT dataset_id, name, source_org, source_url, description, first_ingested_at
                    FROM datasets
                    WHERE source_org = :o
                    ORDER BY first_ingested_at DESC
                    LIMIT 200
                """),
                {"o": org},
            )
        else:
            r = conn.execute(
                text("""
                    SELECT dataset_id, name, source_org, source_url, description, first_ingested_at
                    FROM datasets
                    ORDER BY first_ingested_at DESC
                    LIMIT 200
                """)
            )

        for (id_, name, source_org, source_url, desc, created_at) in r:
            uid = None
            if "/d/" in (source_url or ""):
                # Online dataset (CDC/SAMHSA)
                try:
                    uid = source_url.split("/d/")[1].split("?")[0].split("/")[0]
                except Exception:
                    uid = None
            elif "local/" in (source_url or ""):
                # Local uploaded dataset
                try:
                    uid = source_url.split("local/")[1]
                except Exception:
                    uid = None
            
            s3_key = f"raw/{source_org}/{uid}.parquet" if uid else None
            
            # Check if indexed
            indexed = False
            chunk_count = 0
            if uid and has_chunks_table:
                result = conn.execute(
                    text("""
                        SELECT COUNT(*) 
                        FROM data_chunks 
                        WHERE org = :o AND dataset_uid = :u
                    """),
                    {"o": source_org, "u": uid}
                ).scalar()
                indexed = result > 0
                chunk_count = result
            
            rows.append({
                "id": id_,
                "name": name,
                "org": source_org,
                "source_url": source_url,
                "uid": uid,
                "s3_key": s3_key,
                "created_at": str(created_at),
                "indexed_for_rag": indexed,
                "chunk_count": chunk_count
            })
    
    return {"count": len(rows), "items": rows}


# -----------------------------
# Quick Preview (no ingest)
# -----------------------------
@app.get("/datasets/quick_preview")
async def quick_preview(
    org: str = Query(..., pattern="^(CDC|SAMHSA)$"),
    uid: str = Query(...),
    rows: int = Query(200, ge=1, le=1000000),
):
    """
    Live preview directly from Socrata without ingesting.
    Used by the UI 'Quick Preview' button.
    """
    try:
        data = await socrata.fetch_rows(org, uid, limit=rows)
        return {
            "records": data[:rows],
            "org": org,
            "uid": uid,
            "source": "socrata_live",
            "count": len(data),
        }
    except Exception as e:
        return {"error": str(e), "org": org, "uid": uid}


# -----------------------------
# Preview from MinIO Parquet
# -----------------------------
import os, io, boto3

@app.get("/datasets/preview")
def preview_dataset(
    org: str,
    uid: str,
    rows: int = 5,
    indicator: str | None = None,
    state: str | None = None
):
    """
    Read a Parquet from MinIO and return a preview.
    Used for visualization and data exploration.
    """
    import pandas as pd

    count = rows or 5
    key = f"raw/{org}/{uid}.parquet"
    bucket = os.environ.get("S3_BUCKET", "mh-raw")
    
    # Use AWS S3 if endpoint is not set or set to empty/"aws"/"none"
    endpoint = os.environ.get("S3_ENDPOINT")
    if endpoint and endpoint.lower() not in ["", "aws", "none"]:
        endpoint_url = endpoint
    else:
        endpoint_url = None  # Use AWS S3

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=(os.environ.get("S3_ACCESS_KEY") or os.environ.get("MINIO_ROOT_USER")),
        aws_secret_access_key=(os.environ.get("S3_SECRET_KEY") or os.environ.get("MINIO_ROOT_PASSWORD")),
        region_name=(os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("S3_REGION") or "us-east-1"),
    )
    
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Object not found: s3://{bucket}/{key} ({e})")

    buf = io.BytesIO(obj["Body"].read())
    df = pd.read_parquet(buf)

    # Apply filters if provided
    if indicator and "indicator" in df.columns:
        df = df[df["indicator"] == indicator]
    if state and "state" in df.columns:
        df = df[df["state"] == state]

    # Return the requested number of rows (or all if count is large enough)
    # This ensures full dataset can be loaded for visualization
    sample_df = df.head(count) if count < len(df) else df

    return {
        "org": org,
        "uid": uid,
        "s3_key": key,
        "rows": int(len(df)),
        "total_rows": int(len(df)),
        "cols": list(df.columns),
        "sample": sample_df.to_dict(orient="records"),
        "returned_rows": int(len(sample_df))
    }


# -----------------------------
# Mount feature routers
# -----------------------------
from . import semantic
app.include_router(semantic.router)

from . import answer
app.include_router(answer.router)

app.include_router(interpret.router)


# -----------------------------
# Additional utility endpoints
# -----------------------------
@app.delete("/datasets/{uid}/index")
async def delete_dataset_index(
    uid: str,
    org: str = Query(..., pattern="^(CDC|SAMHSA)$")
):
    """
    Remove index for a specific dataset (useful for re-indexing)
    """
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                DELETE FROM data_chunks 
                WHERE org = :o AND dataset_uid = :u
                RETURNING dataset_uid
            """),
            {"o": org, "u": uid}
        )
        deleted = result.rowcount
    
    return {
        "deleted": deleted > 0,
        "dataset_uid": uid,
        "chunks_removed": deleted
    }


@app.get("/stats")
async def get_system_stats():
    """
    Get overall system statistics
    """
    with engine.begin() as conn:
        # Dataset stats
        dataset_stats = conn.execute(
            text("""
                SELECT 
                    source_org as org,
                    COUNT(*) as count
                FROM datasets
                GROUP BY source_org
            """)
        ).mappings().all()
        
        # Check if tables exist
        has_semantic = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'semantic_index'
                )
            """)
        ).scalar()
        
        has_chunks = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'data_chunks'
                )
            """)
        ).scalar()
        
        semantic_stats = []
        chunk_stats = []
        
        if has_semantic:
            semantic_stats = conn.execute(
                text("""
                    SELECT 
                        org,
                        COUNT(*) as count
                    FROM semantic_index
                    GROUP BY org
                """)
            ).mappings().all()
        
        if has_chunks:
            chunk_stats = conn.execute(
                text("""
                    SELECT 
                        org,
                        COUNT(DISTINCT dataset_uid) as datasets,
                        COUNT(*) as chunks
                    FROM data_chunks
                    GROUP BY org
                """)
            ).mappings().all()
        
        return {
            "datasets": [dict(d) for d in dataset_stats],
            "semantic_index": [dict(s) for s in semantic_stats],
            "data_chunks": [dict(c) for c in chunk_stats],
            "tables": {
                "has_semantic_index": has_semantic,
                "has_data_chunks": has_chunks
            }
        }


# -----------------------------
# Download dataset as JSON
# -----------------------------
@app.get("/download/{org}/{uid}")
async def download_dataset(org: str, uid: str):
    """
    Download a dataset as JSON
    Returns the raw data from the Parquet file
    """
    import boto3
    import os

    try:
        # Use AWS S3 if endpoint is not set or set to empty/"aws"/"none"
        endpoint = settings.S3_ENDPOINT
        if endpoint and endpoint.lower() not in ["", "aws", "none"]:
            endpoint_url = endpoint
        else:
            endpoint_url = None  # Use AWS S3

        # Get S3 client
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.S3_ACCESS_KEY or os.environ.get("MINIO_ROOT_USER"),
            aws_secret_access_key=settings.S3_SECRET_KEY or os.environ.get("MINIO_ROOT_PASSWORD"),
            region_name=settings.AWS_DEFAULT_REGION,
        )

        # Download from S3
        s3_key = f"raw/{org}/{uid}.parquet"
        obj = s3.get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
        buf = io.BytesIO(obj["Body"].read())

        # Read parquet
        df = pd.read_parquet(buf)

        # Convert to JSON
        return {
            "uid": uid,
            "org": org,
            "rows": len(df),
            "columns": len(df.columns),
            "data": json.loads(df.to_json(orient="records", date_format="iso"))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download dataset: {str(e)}")