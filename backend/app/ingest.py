import os
import io
import pandas as pd
import boto3
from botocore.exceptions import ClientError

def _s3_client():
    endpoint = os.environ.get("S3_ENDPOINT", "http://minio:9000")
    # Prefer S3_* keys; fall back to MinIO root creds if not present
    access_key = os.environ.get("S3_ACCESS_KEY") or os.environ.get("MINIO_ROOT_USER")
    secret_key = os.environ.get("S3_SECRET_KEY") or os.environ.get("MINIO_ROOT_PASSWORD")
    region = (os.environ.get("AWS_DEFAULT_REGION")
              or os.environ.get("S3_REGION")
              or "us-east-1")
    
    if not access_key or not secret_key:
        raise RuntimeError("Missing S3 credentials: set S3_ACCESS_KEY/S3_SECRET_KEY or MINIO_ROOT_USER/MINIO_ROOT_PASSWORD")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

def ensure_bucket_exists(bucket_name: str):
    """
    Create bucket if it doesn't exist.
    """
    s3 = _s3_client()
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"[ingest] Bucket {bucket_name} exists")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            try:
                s3.create_bucket(Bucket=bucket_name)
                print(f"[ingest] Created bucket: {bucket_name}")
            except ClientError as create_err:
                print(f"[ingest] Error creating bucket: {create_err}")
                raise
        else:
            print(f"[ingest] Error checking bucket: {e}")
            raise

def write_parquet_to_minio(df: pd.DataFrame, key: str):
    """
    Serialize df to parquet in-memory and upload to MinIO/S3 as `key`
    in the bucket specified by S3_BUCKET.
    """
    bucket = os.environ.get("S3_BUCKET", "mh-raw")
    
    # Ensure bucket exists
    ensure_bucket_exists(bucket)
    
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine='pyarrow')
    buf.seek(0)

    s3 = _s3_client()
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        print(f"[ingest] ✓ Uploaded s3://{bucket}/{key} ({len(df)} rows, {len(buf.getvalue())} bytes)", flush=True)
    except ClientError as e:
        print(f"[ingest] ✗ Error uploading to S3: {e}", flush=True)
        raise

def read_parquet_from_minio(key: str) -> pd.DataFrame:
    """
    Read a Parquet file from MinIO/S3 into a DataFrame.
    """
    bucket = os.environ.get("S3_BUCKET", "mh-raw")
    
    s3 = _s3_client()
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read()
        
        buf = io.BytesIO(data)
        df = pd.read_parquet(buf, engine='pyarrow')
        print(f"[ingest] ✓ Read {len(df)} rows from s3://{bucket}/{key}", flush=True)
        return df
    except ClientError as e:
        print(f"[ingest] ✗ Error reading from S3: {e}", flush=True)
        raise