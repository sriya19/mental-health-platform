# backend/app/baltimore_indexer.py
"""
Baltimore-specific data indexer
Filters datasets to only include Maryland/Baltimore data before indexing
"""
import io
import os
import json
from typing import List, Dict, Any
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

def filter_baltimore_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to only include Baltimore City / Maryland data.
    Handles various column naming conventions.
    """
    original_count = len(df)

    # Convert column names to lowercase for easier matching
    df_lower_cols = {col: col.lower() for col in df.columns}

    # Maryland state-level filters
    md_filters = []

    # Check for state columns
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['state', 'stateabbr', 'state_abbr', 'statedesc', 'state_desc', 'locationabbr', 'locationdesc']:
            # Filter for Maryland
            if 'abbr' in col_lower or col_lower == 'state':
                md_filters.append(df[col].astype(str).str.upper().isin(['MD', 'MARYLAND']))
            else:
                md_filters.append(df[col].astype(str).str.contains('Maryland', case=False, na=False))

    # Check for county/location columns for Baltimore
    baltimore_filters = []
    for col in df.columns:
        col_lower = col.lower()
        if 'county' in col_lower or 'location' in col_lower:
            # Filter for Baltimore
            baltimore_filters.append(df[col].astype(str).str.contains('Baltimore', case=False, na=False))

        # Check for FIPS code (Baltimore City = 24510, Baltimore County = 24005)
        if 'fips' in col_lower or col_lower == 'countyfips':
            baltimore_filters.append(df[col].astype(str).isin(['24005', '24510', '24', '24000']))

    # Combine filters
    combined_filter = None

    # Include data if it's either Maryland state-level OR Baltimore county-level
    if md_filters:
        md_combined = md_filters[0]
        for f in md_filters[1:]:
            md_combined = md_combined | f
        combined_filter = md_combined

    if baltimore_filters:
        balt_combined = baltimore_filters[0]
        for f in baltimore_filters[1:]:
            balt_combined = balt_combined | f

        if combined_filter is not None:
            combined_filter = combined_filter | balt_combined
        else:
            combined_filter = balt_combined

    if combined_filter is not None:
        filtered_df = df[combined_filter]
        print(f"[Baltimore Filter] Filtered {original_count} rows -> {len(filtered_df)} Maryland/Baltimore rows")
        return filtered_df
    else:
        print(f"[Baltimore Filter] No state/location columns found, returning all data")
        return df

async def index_baltimore_data(org: str, uid: str) -> Dict[str, Any]:
    """
    Load dataset, filter for Baltimore/Maryland data only, then index it.
    """
    s3_key = f"raw/{org}/{uid}.parquet"
    bucket = settings.S3_BUCKET

    print(f"[Baltimore Indexer] Processing {org}/{uid}")

    try:
        # Load the parquet file from MinIO
        s3 = _s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf)

        print(f"[Baltimore Indexer] Loaded {len(df)} total rows")

        # Filter for Baltimore/Maryland data
        baltimore_df = filter_baltimore_data(df)

        if len(baltimore_df) == 0:
            print(f"[Baltimore Indexer] No Baltimore/Maryland data found in {uid}")
            return {
                "success": False,
                "dataset_uid": uid,
                "reason": "no_baltimore_data",
                "total_rows": len(df),
                "baltimore_rows": 0
            }

        # Create chunks from Baltimore data
        chunks = await create_baltimore_chunks(baltimore_df, org, uid)

        print(f"[Baltimore Indexer] Created {len(chunks)} Baltimore-specific chunks")

        # Store chunks in database
        with engine.begin() as conn:
            # Remove old chunks for this dataset
            conn.execute(
                text("DELETE FROM data_chunks WHERE org = :o AND dataset_uid = :u"),
                {"o": org, "u": uid}
            )

            # Insert new Baltimore-specific chunks
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

        print(f"[Baltimore Indexer] Successfully indexed {inserted} Baltimore chunks for {uid}")

        return {
            "success": True,
            "dataset_uid": uid,
            "total_rows": len(df),
            "baltimore_rows": len(baltimore_df),
            "chunks_created": inserted,
            "s3_key": s3_key
        }

    except Exception as e:
        print(f"[Baltimore Indexer] Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "dataset_uid": uid
        }

async def create_baltimore_chunks(df: pd.DataFrame, org: str, uid: str) -> List[Dict[str, Any]]:
    """
    Create chunks from Baltimore-specific data.
    Since all data is already Baltimore-specific, we can chunk more aggressively.
    """
    chunks = []
    chunk_size = 50  # Rows per chunk

    # Identify grouping columns
    group_cols = []
    potential_cols = ['indicator', 'year', 'category', 'stratification', 'locationdesc', 'countyname']

    for col in potential_cols:
        if col in [c.lower() for c in df.columns]:
            actual_col = [c for c in df.columns if c.lower() == col][0]
            group_cols.append(actual_col)

    print(f"[Baltimore Chunks] Grouping by: {group_cols[:2] if group_cols else 'sequential chunks'}")

    if len(group_cols) >= 1:
        # Group by indicator/year/category
        try:
            grouped = df.groupby(group_cols[:2] if len(group_cols) >= 2 else group_cols[:1])

            for group_keys, group_df in grouped:
                if len(group_df) == 0:
                    continue

                # Create summary text
                summary_lines = [f"Baltimore/Maryland data for {group_cols[0]}: {group_keys if isinstance(group_keys, str) else group_keys[0]}"]

                # Add statistics
                numeric_cols = group_df.select_dtypes(include=['number']).columns
                for col in numeric_cols[:5]:
                    if group_df[col].notna().any():
                        try:
                            mean_val = group_df[col].mean()
                            min_val = group_df[col].min()
                            max_val = group_df[col].max()
                            summary_lines.append(f"{col}: mean={mean_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
                        except:
                            pass

                # Add sample records
                sample_size = min(3, len(group_df))
                for _, row in group_df.head(sample_size).iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items()
                                           if pd.notna(val) and str(val).strip()][:8])
                    summary_lines.append(f"Sample: {row_text}")

                content = "\n".join(summary_lines)
                summary = content[:500]  # First 500 chars for embedding

                # Generate embedding
                emb = await embed_text(summary)

                chunk_id = f"{uid}_baltimore_{len(chunks)}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "summary": summary,
                    "metadata": {
                        "location": "Baltimore/Maryland",
                        "group_by": dict(zip(group_cols[:2], group_keys)) if isinstance(group_keys, tuple) else {group_cols[0]: group_keys},
                        "row_count": len(group_df)
                    },
                    "embedding": emb
                })

                if len(chunks) >= 100:
                    break

        except Exception as e:
            print(f"[Baltimore Chunks] Grouping failed: {e}, using sequential chunks")
            group_cols = []

    # Fallback: sequential chunking
    if len(chunks) == 0:
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]

            summary_lines = [f"Baltimore/Maryland mental health data (rows {i} to {i+len(chunk_df)})"]

            # Add samples
            for _, row in chunk_df.head(3).iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()
                                       if pd.notna(val)][:8])
                summary_lines.append(row_text)

            content = "\n".join(summary_lines)
            summary = content[:500]

            emb = await embed_text(summary)

            chunk_id = f"{uid}_baltimore_{i//chunk_size}"
            chunks.append({
                "chunk_id": chunk_id,
                "content": content,
                "summary": summary,
                "metadata": {
                    "location": "Baltimore/Maryland",
                    "row_range": [i, min(i+chunk_size, len(df))]
                },
                "embedding": emb
            })

            if len(chunks) >= 100:
                break

    return chunks
