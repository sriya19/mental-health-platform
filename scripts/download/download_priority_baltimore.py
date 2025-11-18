"""
Download and Ingest Priority Baltimore Datasets
- Baltimore Naloxone Data (2 datasets from Open Baltimore)
- Quick win datasets that are immediately downloadable
"""
import requests
import pandas as pd
import io
import time
from pathlib import Path

BACKEND_URL = "http://localhost:8000"

# Priority Baltimore datasets that are immediately downloadable
PRIORITY_DATASETS = [
    {
        "name": "Baltimore Fire Dept - Leave-Behind Naloxone Distribution",
        "description": "BCFD EMS clinicians distribute leave-behind naloxone kits to opioid overdose patients",
        "url": "https://services1.arcgis.com/mVFRs7NF4iFitgbY/arcgis/rest/services/Leave_Behind_Naloxone_Distribution/FeatureServer/0/query",
        "params": {
            "where": "1=1",
            "outFields": "*",
            "f": "json",
            "resultRecordCount": 10000
        },
        "type": "arcgis"
    },
    {
        "name": "Baltimore Fire Dept - Clinician-Administered Naloxone",
        "description": "Records instances where BCFD clinicians administered naloxone for opioid overdoses",
        "url": "https://services1.arcgis.com/mVFRs7NF4iFitgbY/arcgis/rest/services/Clinician_Administered_Naloxone/FeatureServer/0/query",
        "params": {
            "where": "1=1",
            "outFields": "*",
            "f": "json",
            "resultRecordCount": 10000
        },
        "type": "arcgis"
    }
]

def download_arcgis_data(url, params, name):
    """Download data from ArcGIS REST API"""
    try:
        print(f"\n[DOWNLOAD] {name}")
        print(f"           URL: {url}")

        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 200:
            data = response.json()

            if "features" in data:
                # Extract records from GeoJSON features
                records = []
                for feature in data["features"]:
                    record = feature.get("attributes", {})
                    records.append(record)

                df = pd.DataFrame(records)

                if len(df) > 0:
                    print(f"           [SUCCESS] Downloaded {len(df):,} rows, {len(df.columns)} columns")
                    return df
                else:
                    print(f"           [ERROR] No data in response")
                    return None
            else:
                print(f"           [ERROR] Unexpected response format")
                return None
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
            print(f"         [ERROR] Status {response.status_code}: {response.text[:200]}")
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
                print(f"        [SUCCESS] Created {chunks} chunks from {balt_rows:,} Baltimore rows")
                return True, chunks
            else:
                print(f"        [INFO] {result.get('reason', 'No Baltimore data found')}")
                return False, 0
        else:
            print(f"        [ERROR] Status {response.status_code}")
            return False, 0

    except Exception as e:
        print(f"        [ERROR] Failed: {e}")
        return False, 0

def main():
    print("="*80)
    print("DOWNLOAD AND INGEST PRIORITY BALTIMORE DATASETS")
    print("="*80)
    print("\nQuick Win Datasets:")
    print("  - Baltimore Fire Dept Naloxone Administration (2 datasets)")
    print("  - Immediate Baltimore opioid response data")
    print("\n" + "="*80)

    downloaded = []
    ingested = []
    indexed = []

    # Step 1: Download datasets
    print("\n[STEP 1] DOWNLOADING PRIORITY DATASETS")
    print("="*80)

    for dataset_config in PRIORITY_DATASETS:
        if dataset_config["type"] == "arcgis":
            df = download_arcgis_data(
                dataset_config["url"],
                dataset_config["params"],
                dataset_config["name"]
            )

            if df is not None:
                downloaded.append({
                    "name": dataset_config["name"],
                    "description": dataset_config["description"],
                    "df": df
                })

        time.sleep(2)  # Rate limiting

    print(f"\n[INFO] Successfully downloaded {len(downloaded)} datasets")

    # Step 2: Upload to backend
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

        time.sleep(2)  # Rate limiting

    print(f"\n[INFO] Successfully ingested {len(ingested)} datasets")

    # Step 3: Index for RAG
    print("\n[STEP 3] INDEXING BALTIMORE DATA FOR RAG")
    print("="*80)

    total_chunks = 0

    for dataset in ingested:
        # Custom org is "Custom", uid is the returned uid
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

        time.sleep(2)  # Rate limiting

    # Summary
    print("\n" + "="*80)
    print("PRIORITY BALTIMORE DATA INGESTION COMPLETE")
    print("="*80)
    print(f"\n[SUMMARY]")
    print(f"  Datasets downloaded: {len(downloaded)}")
    print(f"  Datasets ingested: {len(ingested)}")
    print(f"  Datasets indexed: {len(indexed)}")
    print(f"  Total new chunks: {total_chunks}")

    if len(indexed) > 0:
        print(f"\n[NEW BALTIMORE DATASETS]")
        for ds in indexed:
            print(f"  - {ds['name']}")
            print(f"    Rows: {ds['rows']:,} | Chunks: {ds['chunks']}")

    print("\n[SUCCESS] Baltimore naloxone data is now searchable!")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
