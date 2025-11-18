"""
Export 10 datasets to JSON from running MinIO container
"""
import subprocess
import json
import pandas as pd
from pathlib import Path
import tempfile
import os

def list_datasets_from_minio():
    """List all datasets in MinIO"""
    cmd = 'docker exec minio mc ls local/mh-raw/raw/'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[INFO] mc not available, trying direct S3 API...")
        # List organizations
        cmd = 'docker exec minio ls /data/mh-raw/raw/'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Could not list MinIO contents")
            return []

    lines = result.stdout.strip().split('\n')
    orgs = [line.split()[-1].strip('/') for line in lines if line.strip()]

    datasets = []
    for org in orgs:
        if not org:
            continue

        cmd = f'docker exec minio ls /data/mh-raw/raw/{org}/'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            files = result.stdout.strip().split('\n')
            for file_line in files:
                if '.parquet' in file_line:
                    filename = file_line.split()[-1]
                    uid = filename.replace('.parquet', '').strip()
                    datasets.append({"org": org, "uid": uid, "filename": filename})

    return datasets

def export_dataset_from_minio(org: str, uid: str, output_dir: Path):
    """Export a dataset from MinIO to JSON"""
    try:
        print(f"\n[EXPORT] {org}/{uid}")

        # Create temporary file for parquet
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Copy from MinIO container to temp file
            source_path = f"/data/mh-raw/raw/{org}/{uid}.parquet"
            cmd = f'docker cp minio:{source_path} "{tmp_path}"'

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"         [ERROR] Could not copy from MinIO: {result.stderr}")
                return {"success": False, "error": "copy_failed"}

            # Check if it's a directory (MinIO metadata)
            if os.path.isdir(tmp_path):
                print(f"         [SKIP] File is MinIO metadata directory")
                return {"success": False, "error": "metadata_dir"}

            # Read parquet
            df = pd.read_parquet(tmp_path)

            print(f"         Rows: {len(df):,} | Columns: {len(df.columns)}")

            # Check for Baltimore data
            baltimore_info = has_baltimore_data(df)

            if baltimore_info["is_baltimore_dataset"]:
                print(f"         [BALTIMORE DATA FOUND!]")
                print(f"         Maryland: {baltimore_info['maryland_rows']:,} | Baltimore: {baltimore_info['baltimore_rows']:,}")

            # Create JSON
            json_filename = f"{org}_{uid}.json"
            json_path = output_dir / json_filename

            data_dict = {
                "metadata": {
                    "dataset_name": f"{org}:{uid}",
                    "uid": uid,
                    "organization": org,
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "columns": list(df.columns),
                    "baltimore_info": baltimore_info
                },
                "data": json.loads(df.to_json(orient="records", date_format="iso"))
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)

            file_size = json_path.stat().st_size / (1024 * 1024)
            print(f"         [SUCCESS] Exported: {json_filename} ({file_size:.2f} MB)")

            return {
                "success": True,
                "org": org,
                "uid": uid,
                "filename": json_filename,
                "rows": len(df),
                "baltimore_info": baltimore_info
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    if os.path.isfile(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass

    except Exception as e:
        print(f"         [ERROR] Failed: {e}")
        return {"success": False, "error": str(e)}

def has_baltimore_data(df: pd.DataFrame) -> dict:
    """Check if dataset contains Baltimore/Maryland data"""
    cols_lower = [col.lower() for col in df.columns]

    state_cols = ['state', 'stateabbr', 'state_abbr', 'statedesc', 'state_desc', 'locationabbr', 'locationdesc']
    has_state = any(col in cols_lower for col in state_cols)

    location_cols = ['county', 'location', 'countyname', 'locationdesc', 'fips', 'countyfips']
    has_location = any(col in cols_lower for col in location_cols)

    md_count = 0
    baltimore_count = 0

    if has_state:
        for col in df.columns:
            if col.lower() in state_cols:
                try:
                    md_count = int(df[col].astype(str).str.contains('MD|Maryland', case=False, na=False, regex=True).sum())
                    if md_count > 0:
                        break
                except:
                    pass

    if has_location:
        for col in df.columns:
            if any(loc in col.lower() for loc in location_cols):
                try:
                    baltimore_count = int(df[col].astype(str).str.contains('Baltimore', case=False, na=False).sum())
                    if baltimore_count > 0:
                        break
                except:
                    pass

    return {
        "has_state_col": has_state,
        "has_location_col": has_location,
        "maryland_rows": md_count,
        "baltimore_rows": baltimore_count,
        "is_baltimore_dataset": baltimore_count > 0 or md_count > 0
    }

def main():
    print("="*80)
    print("EXPORTING 10 DATASETS TO JSON FORMAT FROM MINIO")
    print("="*80)

    print("\n[INFO] Listing datasets from MinIO container...")
    datasets = list_datasets_from_minio()

    if not datasets:
        print("[ERROR] No datasets found in MinIO!")
        return

    print(f"[INFO] Found {len(datasets)} datasets in MinIO")

    # Prioritize BRFSS (likely has state/county data)
    brfss = [d for d in datasets if d["org"].upper() == "BRFSS"]
    others = [d for d in datasets if d["org"].upper() != "BRFSS"]

    sorted_datasets = brfss + others

    # Select top 10
    selected = sorted_datasets[:10]

    print(f"[INFO] Selected 10 datasets for export")
    print(f"[INFO] {len([d for d in selected if d['org'].upper() == 'BRFSS'])} BRFSS datasets (likely to have Baltimore data)\n")

    # Create output directory
    output_dir = Path("team-datasets-json")
    output_dir.mkdir(exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}/\n")

    print("="*80)
    print("STARTING EXPORT")
    print("="*80)

    results = []
    baltimore_count = 0

    for ds in selected:
        result = export_dataset_from_minio(ds["org"], ds["uid"], output_dir)
        results.append(result)

        if result.get("success") and result.get("baltimore_info", {}).get("is_baltimore_dataset"):
            baltimore_count += 1

    # Summary
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)

    successful = sum(1 for r in results if r.get("success"))
    print(f"\n[SUCCESS] Exported {successful} out of {len(selected)} datasets")
    print(f"[INFO] Datasets with Baltimore/Maryland data: {baltimore_count}")

    json_files = list(output_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in json_files) / (1024 * 1024)

    print(f"\n[SUMMARY]")
    print(f"  Files exported: {len(json_files)}")
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Baltimore datasets: {baltimore_count}")
    print(f"\n[SUCCESS] Files ready in: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
