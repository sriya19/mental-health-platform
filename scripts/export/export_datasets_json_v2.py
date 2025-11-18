"""
Export 10 datasets to JSON format for team sharing
Runs inside Docker backend container
"""
import os
import io
import json
import pandas as pd
import boto3
from pathlib import Path
from sqlalchemy import create_engine, text

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app_user:app_password@pg:5432/mh_catalog")

# MinIO configuration
MINIO_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
S3_BUCKET = "mh-raw"

def get_s3_client():
    """Get MinIO S3 client"""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1"
    )

def get_datasets_from_db():
    """Get all datasets from database"""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT uid, name, organization, description
            FROM datasets
            WHERE uid IS NOT NULL AND uid != 'None'
            ORDER BY created_at DESC
            LIMIT 40
        """))
        return [dict(row._mapping) for row in result]

def has_baltimore_data(df: pd.DataFrame) -> dict:
    """
    Check if dataset contains Baltimore/Maryland data
    Returns dict with info about Baltimore data presence
    """
    # Convert column names to lowercase for matching
    cols_lower = [col.lower() for col in df.columns]

    # Check for state columns
    state_cols = ['state', 'stateabbr', 'state_abbr', 'statedesc', 'state_desc', 'locationabbr', 'locationdesc']
    has_state = any(col in cols_lower for col in state_cols)

    # Check for location/county columns
    location_cols = ['county', 'location', 'countyname', 'locationdesc', 'fips', 'countyfips']
    has_location = any(col in cols_lower for col in location_cols)

    # Try to count Maryland data
    md_count = 0
    baltimore_count = 0

    if has_state:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in state_cols:
                try:
                    md_count = df[col].astype(str).str.contains('MD|Maryland', case=False, na=False, regex=True).sum()
                    if md_count > 0:
                        break
                except:
                    pass

    if has_location:
        for col in df.columns:
            col_lower = col.lower()
            if any(loc in col_lower for loc in location_cols):
                try:
                    baltimore_count = df[col].astype(str).str.contains('Baltimore', case=False, na=False).sum()
                    if baltimore_count > 0:
                        break
                except:
                    pass

    return {
        "has_state_col": has_state,
        "has_location_col": has_location,
        "maryland_rows": int(md_count),
        "baltimore_rows": int(baltimore_count),
        "is_baltimore_dataset": baltimore_count > 0 or md_count > 0
    }

def export_dataset_to_json(org: str, uid: str, dataset_name: str, output_dir: Path):
    """
    Export a dataset from MinIO to JSON format
    """
    s3_key = f"raw/{org}/{uid}.parquet"

    try:
        print(f"\n[EXPORT] {dataset_name}")
        print(f"         UID: {uid}")

        # Load parquet from MinIO
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf)

        print(f"         Rows: {len(df):,}")

        # Check for Baltimore data
        baltimore_info = has_baltimore_data(df)

        if baltimore_info["is_baltimore_dataset"]:
            print(f"         [BALTIMORE DATA FOUND!]")
            print(f"         Maryland rows: {baltimore_info['maryland_rows']:,}")
            print(f"         Baltimore rows: {baltimore_info['baltimore_rows']:,}")

        # Create safe filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in dataset_name)
        safe_name = safe_name[:50]  # Limit length
        json_filename = f"{safe_name}_{uid}.json"
        json_path = output_dir / json_filename

        # Convert to JSON
        data_dict = {
            "metadata": {
                "dataset_name": dataset_name,
                "uid": uid,
                "organization": org,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "baltimore_info": baltimore_info
            },
            "data": json.loads(df.to_json(orient="records", date_format="iso"))
        }

        # Write JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

        file_size = os.path.getsize(json_path) / (1024 * 1024)
        print(f"         [SUCCESS] Exported: {json_filename} ({file_size:.2f} MB)")

        return {
            "success": True,
            "dataset_name": dataset_name,
            "uid": uid,
            "filename": json_filename,
            "rows": len(df),
            "baltimore_info": baltimore_info
        }

    except Exception as e:
        print(f"         [ERROR] Failed: {e}")
        return {
            "success": False,
            "dataset_name": dataset_name,
            "uid": uid,
            "error": str(e)
        }

def main():
    print("="*80)
    print("EXPORTING 10 DATASETS TO JSON FORMAT")
    print("="*80)

    # Create output directory
    output_dir = Path("/app/team-datasets-json")
    output_dir.mkdir(exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}/\n")

    # Get all datasets from database
    try:
        datasets = get_datasets_from_db()
        print(f"[INFO] Found {len(datasets)} total datasets in database")

        # Sort to prioritize datasets that might have Baltimore data
        baltimore_keywords = ['state', 'county', 'location', 'geographic', 'maryland', 'baltimore', 'brfss', 'behavioral']

        def score_dataset(ds):
            """Score dataset by likelihood of having Baltimore data"""
            score = 0
            name = ds.get('name', '').lower()
            desc = ds.get('description', '').lower() if ds.get('description') else ''

            for keyword in baltimore_keywords:
                if keyword in name or keyword in desc:
                    score += 1

            return score

        # Sort by score (descending)
        datasets_sorted = sorted(datasets, key=score_dataset, reverse=True)

        # Select top 10 datasets
        selected_datasets = datasets_sorted[:10]

        print(f"[INFO] Selected 10 datasets for export:\n")
        for i, ds in enumerate(selected_datasets, 1):
            print(f"  {i}. {ds.get('name', 'Unknown')[:70]}")

        print("\n" + "="*80)
        print("STARTING EXPORT")
        print("="*80)

        # Export each dataset
        results = []
        baltimore_count = 0

        for ds in selected_datasets:
            org = ds.get('organization', 'Unknown')
            uid = ds.get('uid', '')
            name = ds.get('name', 'Unknown Dataset')

            if not uid:
                print(f"\n[SKIP] {name} - No UID")
                continue

            result = export_dataset_to_json(org, uid, name, output_dir)
            results.append(result)

            if result.get("success") and result.get("baltimore_info", {}).get("is_baltimore_dataset"):
                baltimore_count += 1

        # Summary
        print("\n" + "="*80)
        print("EXPORT COMPLETE")
        print("="*80)

        successful = sum(1 for r in results if r.get("success"))
        print(f"\n[SUCCESS] Exported {successful} out of {len(selected_datasets)} datasets")
        print(f"[INFO] Datasets with Baltimore/Maryland data: {baltimore_count}")
        print(f"[INFO] Output directory: {output_dir}/")

        # List files
        json_files = list(output_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in json_files) / (1024 * 1024)

        print(f"\n[SUMMARY]")
        print(f"  Files exported: {len(json_files)}")
        print(f"  Total size: {total_size:.2f} MB")
        print(f"  Baltimore datasets: {baltimore_count}")
        print("\n[SUCCESS] Files will be copied to host system...")
        print("="*80)

    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
