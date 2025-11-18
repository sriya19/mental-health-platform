# backend/app/rag.py
import io
import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import boto3
from sqlalchemy import text
from .db import engine
from .config import settings
from .semantic import embed_text

def _s3_client():
    """Get S3/MinIO client"""
    # Use AWS S3 if endpoint is not set or set to empty/"aws"/"none"
    endpoint = settings.S3_ENDPOINT
    if endpoint and endpoint.lower() not in ["", "aws", "none"]:
        endpoint_url = endpoint
    else:
        endpoint_url = None  # Use AWS S3

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.S3_ACCESS_KEY or os.environ.get("MINIO_ROOT_USER"),
        aws_secret_access_key=settings.S3_SECRET_KEY or os.environ.get("MINIO_ROOT_PASSWORD"),
        region_name=settings.AWS_DEFAULT_REGION,
    )

def _vec_literal(vec: list[float]) -> str:
    """Convert Python list -> SQL vector literal for pgvector."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

async def index_dataset_content(org: str, uid: str, limit_rows: int = 20000) -> Dict[str, Any]:
    """
    Load a dataset from MinIO and create searchable chunks from actual data.
    This processes the real data rows, not just metadata.
    """
    s3_key = f"raw/{org}/{uid}.parquet"
    bucket = settings.S3_BUCKET
    
    print(f"[RAG] Indexing dataset {org}/{uid} from {s3_key}")
    
    try:
        # Load the parquet file from MinIO
        s3 = _s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf)
        
        print(f"[RAG] Loaded {len(df)} rows from {uid}")
        
        if limit_rows and len(df) > limit_rows:
            df = df.head(limit_rows)
        
        # Create meaningful chunks from the data
        chunks = await create_data_chunks(df, org, uid)
        
        print(f"[RAG] Created {len(chunks)} chunks")
        
        # Store chunks in database with embeddings
        with engine.begin() as conn:
            # Create table if it doesn't exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS data_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    org TEXT NOT NULL,
                    dataset_uid TEXT NOT NULL,
                    chunk_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    metadata JSONB,
                    embedding vector(1536),
                    created_at TIMESTAMPTZ DEFAULT now()
                );"""))
            conn.execute(text("""CREATE INDEX IF NOT EXISTS data_chunks_embedding_idx 
                ON data_chunks USING ivfflat (embedding vector_cosine_ops);"""))
            conn.execute(text("""CREATE INDEX IF NOT EXISTS data_chunks_org_uid_idx 
                ON data_chunks(org, dataset_uid);
            """))
            
            # Remove old chunks for this dataset
            conn.execute(
                text("DELETE FROM data_chunks WHERE org = :o AND dataset_uid = :u"),
                {"o": org, "u": uid}
            )
            
            # Insert new chunks
            # Insert new chunks
            inserted = 0
            for chunk in chunks:
                meta_json = json.dumps(chunk.get("metadata", {}))
                emb_vector = _vec_literal(chunk["embedding"])
    
                conn.execute(
                    text(f"""
                        INSERT INTO data_chunks 
                        (org, dataset_uid, chunk_id, content, summary, metadata, embedding)
                        VALUES (:o, :u, :cid, :content, :summary, '{meta_json}'::jsonb, '{emb_vector}'::vector)
                    """),
                    {
                        "o": org,
                        "u": uid,
                        "cid": chunk["chunk_id"],
                        "content": chunk["content"],
                        "summary": chunk.get("summary", "")
                    }
                )
                inserted += 1
        
        print(f"[RAG] Successfully indexed {inserted} chunks for {uid}")
        
        return {
            "success": True,
            "dataset_uid": uid,
            "rows_processed": len(df),
            "chunks_created": inserted,
            "s3_key": s3_key
        }
        
    except Exception as e:
        print(f"[RAG] Error indexing {uid}: {e}")
        return {
            "success": False,
            "error": str(e),
            "dataset_uid": uid
        }

async def create_data_chunks(df: pd.DataFrame, org: str, uid: str, 
                            chunk_size: int = 50) -> List[Dict[str, Any]]:
    """
    Convert DataFrame rows into searchable chunks with embeddings.
    Groups rows intelligently based on common attributes.
    """
    chunks = []
    
    # Identify key columns for grouping
    group_cols = []
    potential_group_cols = ['state', 'location', 'locationdesc', 'year', 'indicator', 
                            'category', 'stratification', 'month', 'period']
    
    for col in potential_group_cols:
        if col in [c.lower() for c in df.columns]:
            # Find the actual column name (case-insensitive match)
            actual_col = [c for c in df.columns if c.lower() == col][0]
            group_cols.append(actual_col)
    
    print(f"[RAG] Grouping by columns: {group_cols}")
    
    if group_cols:
        # Group by available columns and create chunks
        try:
            grouped = df.groupby(group_cols[:2])  # Limit to 2 columns to avoid too many groups
            
            for group_keys, group_df in grouped:
                if len(group_df) == 0:
                    continue
                    
                # Create chunk content
                chunk_text, summary = create_chunk_text(group_df, group_cols[:2], group_keys)
                
                # Create metadata
                metadata = {
                    "group_by": dict(zip(group_cols[:2], group_keys)) if isinstance(group_keys, tuple) else {group_cols[0]: group_keys},
                    "row_count": len(group_df),
                    "columns": list(group_df.columns)[:20]  # Limit columns in metadata
                }
                
                # Generate embedding for the summary
                summary_to_embed = summary.strip() if summary and summary.strip() else (
                    "No numeric or categorical data summary available."
                )
                emb = await embed_text(summary_to_embed)
                
                chunk_id = f"{uid}_{len(chunks)}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "summary": summary,
                    "metadata": metadata,
                    "embedding": emb
                })
                
                # Limit total chunks to avoid excessive processing
                if len(chunks) >= 100:
                    print(f"[RAG] Reached chunk limit of 100")
                    break
                    
        except Exception as e:
            print(f"[RAG] Error grouping data: {e}. Creating sequential chunks instead.")
            # Fall back to sequential chunking
            group_cols = []
    
    if not group_cols or len(chunks) == 0:
        # No grouping columns or grouping failed, create chunks from sequential rows
        print(f"[RAG] Creating sequential chunks")
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            chunk_text, summary = create_chunk_text(chunk_df, [], [])
            
            # Generate embedding
            emb = await embed_text(summary)
            
            chunk_id = f"{uid}_{i//chunk_size}"
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_text,
                "summary": summary,
                "metadata": {
                    "row_range": [i, min(i+chunk_size, len(df))],
                    "columns": list(chunk_df.columns)[:20]
                },
                "embedding": emb
            })
            
            if len(chunks) >= 100:
                break
    
    return chunks

def create_chunk_text(df: pd.DataFrame, group_cols: List[str], 
                      group_keys: tuple) -> tuple[str, str]:
    """
    Convert a DataFrame chunk into searchable text.
    Returns (full_content, summary_for_embedding)
    """
    lines = []
    summary_lines = []
    
    # Add group information if available
    if group_cols and group_keys:
        if isinstance(group_keys, tuple):
            group_info = ", ".join([f"{col}: {val}" for col, val in zip(group_cols, group_keys)])
        else:
            group_info = f"{group_cols[0]}: {group_keys}"
        lines.append(f"Data for {group_info}")
        summary_lines.append(f"Data for {group_info}")
        lines.append("")
    
    # Add summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    stats_added = 0
    for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
        if df[col].notna().any():
            try:
                stats = df[col].describe()
                stat_line = f"{col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}, count={stats['count']:.0f}"
                lines.append(stat_line)
                if stats_added < 3:  # Only first 3 for summary
                    summary_lines.append(stat_line)
                    stats_added += 1
            except:
                pass
    
    # Add categorical value counts for non-numeric columns
    cat_cols = df.select_dtypes(exclude=['number']).columns
    cat_added = 0
    for col in cat_cols[:5]:  # Limit to first 5 categorical columns
        if col not in group_cols:  # Skip grouping columns
            try:
                value_counts = df[col].value_counts().head(5)
                if not value_counts.empty:
                    values = ", ".join([f"{val}: {cnt}" for val, cnt in value_counts.items()])
                    cat_line = f"{col} distribution: {values}"
                    lines.append(cat_line)
                    if cat_added < 2:  # Only first 2 for summary
                        summary_lines.append(cat_line)
                        cat_added += 1
            except:
                pass
    
    # Add sample rows for context
    lines.append("\nSample records:")
    sample_size = min(5, len(df))
    for idx, (_, row) in enumerate(df.head(sample_size).iterrows()):
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() 
                               if pd.notna(val) and str(val).strip()][:10])  # Limit fields
        lines.append(row_text)
        if idx < 2:  # Only first 2 rows for summary
            summary_lines.append(f"Sample: {row_text[:100]}...")
    
    full_content = "\n".join(lines)
    summary = "\n".join(summary_lines)[:1000]  # Limit summary length
    
    return full_content, summary

async def search_dataset_chunks(question: str, org: str = "CDC", k: int = 10) -> List[Dict[str, Any]]:
    """
    Search through actual dataset chunks stored in the database.
    Returns chunks of actual data that are most relevant to the question.
    Prioritizes Baltimore/Maryland data when location keywords are detected.
    """
    # Embed the question
    emb = await embed_text(question)
    vec = _vec_literal(emb)

    # Detect location-specific queries
    question_lower = question.lower()
    baltimore_keywords = ['baltimore', 'maryland', 'md']
    is_baltimore_query = any(keyword in question_lower for keyword in baltimore_keywords)

    with engine.begin() as conn:
        # First, check if we have any data chunks
        if org == "All":
            result = conn.execute(
                text("SELECT COUNT(*) as cnt FROM data_chunks")
            ).fetchone()
        else:
            result = conn.execute(
                text("SELECT COUNT(*) as cnt FROM data_chunks WHERE org = :o"),
                {"o": org}
            ).fetchone()

        if result[0] == 0:
            print(f"[RAG] No data chunks found for {org}")
            return []

        print(f"[RAG] Searching {result[0]} data chunks for: {question[:50]}...")
        if is_baltimore_query:
            print(f"[RAG] Baltimore-specific query detected - filtering for Maryland/Baltimore data")

        # If Baltimore query, first try to get Baltimore-specific chunks
        if is_baltimore_query:
            if org == "All":
                baltimore_rows = conn.execute(
                    text("""
                        SELECT
                            dc.chunk_id,
                            dc.dataset_uid,
                            dc.org,
                            dc.content,
                            dc.summary,
                            dc.metadata,
                            d.name as dataset_name,
                            d.description as dataset_description
                        FROM data_chunks dc
                        LEFT JOIN datasets d ON d.source_url LIKE '%' || dc.dataset_uid || '%'
                        WHERE (
                              dc.content ILIKE '%baltimore%'
                              OR dc.content ILIKE '%maryland%'
                              OR dc.content ILIKE '% MD %'
                              OR dc.content ILIKE '%countyfips: 24005%'
                              OR dc.content ILIKE '%stateabbr: MD%'
                          )
                        ORDER BY dc.embedding <-> CAST(:v AS vector)
                        LIMIT :k
                    """),
                    {"v": vec, "k": k}
                ).mappings().all()
            else:
                baltimore_rows = conn.execute(
                    text("""
                        SELECT
                            dc.chunk_id,
                            dc.dataset_uid,
                            dc.content,
                            dc.summary,
                            dc.metadata,
                            d.name as dataset_name,
                            d.description as dataset_description
                        FROM data_chunks dc
                        LEFT JOIN datasets d ON d.source_url LIKE '%' || dc.dataset_uid || '%'
                        WHERE dc.org = :o
                          AND (
                              dc.content ILIKE '%baltimore%'
                              OR dc.content ILIKE '%maryland%'
                              OR dc.content ILIKE '% MD %'
                              OR dc.content ILIKE '%countyfips: 24005%'
                              OR dc.content ILIKE '%stateabbr: MD%'
                          )
                        ORDER BY dc.embedding <-> CAST(:v AS vector)
                        LIMIT :k
                    """),
                    {"o": org, "v": vec, "k": k}
                ).mappings().all()

            if baltimore_rows:
                print(f"[RAG] Found {len(baltimore_rows)} Baltimore-specific data chunks")
                return [dict(r) for r in baltimore_rows]
            else:
                print(f"[RAG] No Baltimore-specific chunks found, falling back to general search")

        # General search (or fallback if no Baltimore data)
        if org == "All":
            rows = conn.execute(
                text("""
                    SELECT
                        dc.chunk_id,
                        dc.dataset_uid,
                        dc.org,
                        dc.content,
                        dc.summary,
                        dc.metadata,
                        d.name as dataset_name,
                        d.description as dataset_description
                    FROM data_chunks dc
                    LEFT JOIN datasets d ON d.source_url LIKE '%' || dc.dataset_uid || '%'
                    ORDER BY dc.embedding <-> CAST(:v AS vector)
                    LIMIT :k
                """),
                {"v": vec, "k": k}
            ).mappings().all()
        else:
            rows = conn.execute(
                text("""
                    SELECT
                        dc.chunk_id,
                        dc.dataset_uid,
                        dc.content,
                        dc.summary,
                        dc.metadata,
                        d.name as dataset_name,
                        d.description as dataset_description
                    FROM data_chunks dc
                    LEFT JOIN datasets d ON d.source_url LIKE '%' || dc.dataset_uid || '%'
                    WHERE dc.org = :o
                    ORDER BY dc.embedding <-> CAST(:v AS vector)
                    LIMIT :k
                """),
                {"o": org, "v": vec, "k": k}
            ).mappings().all()

        print(f"[RAG] Found {len(rows)} relevant data chunks")
        return [dict(r) for r in rows]