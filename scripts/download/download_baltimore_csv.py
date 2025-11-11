"""
Download Baltimore Datasets - Direct CSV Export
Uses CSV export URLs from Open Baltimore ArcGIS Hub
"""
import requests
import pandas as pd
import io
import time

BACKEND_URL = "http://localhost:8000"

# Baltimore datasets with direct CSV export URLs
BALTIMORE_CSV_DATASETS = [
    {
        "name": "Baltimore Fire Dept - Leave-Behind Naloxone",
        "description": "BCFD EMS leave-behind naloxone kit distribution to opioid overdose patients",
        "csv_url": "https://opendata.arcgis.com/api/v3/datasets/2e5f8e26e40f4f1db0b96f1228b0b7ed_0/downloads/data?format=csv&spatialRefId=4326"
    },
    {
        "name": "Baltimore Fire Dept - Administered Naloxone",
        "description": "BCFD clinician-administered naloxone for opioid overdoses",
        "csv_url": "https://opendata.arcgis.com/api/v3/datasets/f3f5afeecf0940cb94088069e2e2d2e4_0/downloads/data?format=csv&spatialRefId=4326"
    }
]

def download_csv(url, name):
    """Download CSV from URL"""
    try:
        print(f"\n[DOWNLOAD] {name}")
        print(f"           URL: {url[:80]}...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=120)

        if response.status_code == 200:
            # Parse CSV
            df = pd.read_csv(io.StringIO(response.text))

            print(f"           [SUCCESS] Downloaded {len(df):,} rows, {len(df.columns)} columns")
            return df
        else:
            print(f"           [ERROR] Status {response.status_code}")
            return None

    except Exception as e:
        print(f"           [ERROR] Download failed: {e}")
        return None

def upload_csv_to_backend(df, dataset_name, description):
    """Upload DataFrame as CSV to backend"""
    try:
        print(f"\n[UPLOAD] {dataset_name}")
        print(f"         Rows: {len(df):,} | Columns: {len(df.columns)}")
        print(f"         Sample columns: {', '.join(df.columns[:5].tolist())}")

        # Convert DataFrame to CSV
        csv_data = df.to_csv(index=False)

        # Upload to backend
        files = {
            'file': ('baltimore_data.csv', io.StringIO(csv_data), 'text/csv')
        }
        data = {
            'dataset_name': dataset_name,
            'description': description
        }

        response = requests.post(
            f"{BACKEND_URL}/upload_csv",
            files=files,
            data=data,
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            uid = result.get('uid', 'N/A')
            print(f"         [SUCCESS] Ingested as UID: {uid}")
            return True, uid
        else:
            print(f"         [ERROR] Status {response.status_code}")
            if response.text:
                print(f"         Response: {response.text[:200]}")
            return False, None

    except Exception as e:
        print(f"         [ERROR] Upload failed: {e}")
        return False, None

def index_baltimore_data(org, uid, name):
    """Index Baltimore data for RAG"""
    try:
        print(f"\n[INDEX] {name}")

        response = requests.post(
            f"{BACKEND_URL}/index_baltimore",
            json={
                "org": org,
                "uid": uid
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()

            if result.get("success"):
                chunks = result.get("chunks_created", 0)
                balt_rows = result.get("baltimore_rows", 0)
                total_rows = result.get("total_rows", 0)
                print(f"        [SUCCESS] {balt_rows:,}/{total_rows:,} Baltimore rows")
                print(f"        Created {chunks} searchable chunks")
                return True, chunks
            else:
                reason = result.get("reason", "Unknown")
                print(f"        [INFO] {reason}")
                return False, 0
        else:
            print(f"        [ERROR] Status {response.status_code}")
            return False, 0

    except Exception as e:
        print(f"        [ERROR] Failed: {e}")
        return False, 0

def main():
    print("="*80)
    print("DOWNLOAD BALTIMORE DATASETS - CSV EXPORT")
    print("="*80)
    print("\nTarget Datasets:")
    for ds in BALTIMORE_CSV_DATASETS:
        print(f"  - {ds['name']}")
    print("\n" + "="*80)

    downloaded = []
    ingested = []
    indexed = []
    total_chunks = 0

    # Step 1: Download CSV files
    print("\n[STEP 1] DOWNLOADING CSV FILES")
    print("="*80)

    for dataset_config in BALTIMORE_CSV_DATASETS:
        df = download_csv(dataset_config["csv_url"], dataset_config["name"])

        if df is not None:
            downloaded.append({
                "name": dataset_config["name"],
                "description": dataset_config["description"],
                "df": df
            })

        time.sleep(2)

    print(f"\n[INFO] Successfully downloaded {len(downloaded)}/{len(BALTIMORE_CSV_DATASETS)} datasets")

    # Step 2: Upload to backend
    if len(downloaded) > 0:
        print("\n[STEP 2] UPLOADING TO BACKEND")
        print("="*80)

        for dataset in downloaded:
            success, uid = upload_csv_to_backend(
                dataset["df"],
                dataset["name"],
                dataset["description"]
            )

            if success:
                ingested.append({
                    "name": dataset["name"],
                    "uid": uid,
                    "rows": len(dataset["df"])
                })

            time.sleep(2)

        print(f"\n[INFO] Successfully ingested {len(ingested)}/{len(downloaded)} datasets")

        # Step 3: Index for RAG
        if len(ingested) > 0:
            print("\n[STEP 3] INDEXING FOR RAG SEARCH")
            print("="*80)

            for dataset in ingested:
                success, chunks = index_baltimore_data(
                    "Custom",
                    dataset["uid"],
                    dataset["name"]
                )

                if success:
                    indexed.append({
                        "name": dataset["name"],
                        "chunks": chunks,
                        "rows": dataset["rows"]
                    })
                    total_chunks += chunks

                time.sleep(2)

    # Summary
    print("\n" + "="*80)
    print("BALTIMORE DATASET INGESTION COMPLETE")
    print("="*80)
    print(f"\n[SUMMARY]")
    print(f"  Datasets downloaded: {len(downloaded)}/{len(BALTIMORE_CSV_DATASETS)}")
    print(f"  Datasets ingested: {len(ingested)}")
    print(f"  Datasets indexed: {len(indexed)}")
    print(f"  Total new chunks: {total_chunks}")

    if len(indexed) > 0:
        print(f"\n[NEW BALTIMORE DATASETS AVAILABLE FOR SEARCH]")
        for ds in indexed:
            print(f"  - {ds['name']}")
            print(f"    Rows: {ds['rows']:,} | Chunks: {ds['chunks']}")
        print(f"\n[SUCCESS] {total_chunks} new Baltimore opioid data chunks ready for Q&A!")
    else:
        print(f"\n[INFO] No new datasets were successfully ingested")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
