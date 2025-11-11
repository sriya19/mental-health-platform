"""
Download and Ingest Baltimore Datasets from Multiple Sources
- BNIA Vital Signs Data (2000-2010)
- SAMHSA Baltimore-Towson Data
- Additional Baltimore health datasets
"""
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path
import time

BACKEND_URL = "http://localhost:8000"

# Known downloadable Baltimore datasets
BALTIMORE_DATA_SOURCES = [
    {
        "name": "BNIA Vital Signs Baltimore 2000-2010",
        "url": "https://bniajfi.org/wp-content/uploads/2020/04/Vital-Signs-10-Data-Tables.zip",
        "type": "zip",
        "description": "Baltimore Neighborhood Indicators Alliance community health indicators 2000-2010"
    },
    {
        "name": "SAMHSA Baltimore-Towson Metro Brief",
        "url": "https://www.samhsa.gov/data/sites/default/files/NSDUHMetroBriefReports/NSDUHMetroBriefReports/NSDUH-Metro-Baltimore.pdf",
        "type": "pdf",
        "description": "NSDUH 2005-2010 substance use and mental health data for Baltimore-Towson MSA"
    }
]

def download_file(url, name):
    """Download a file from URL"""
    try:
        print(f"\n[DOWNLOAD] {name}")
        print(f"           URL: {url}")

        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            size = len(response.content) / (1024 * 1024)
            print(f"           [SUCCESS] Downloaded {size:.2f} MB")
            return response.content
        else:
            print(f"           [ERROR] Status {response.status_code}")
            return None

    except Exception as e:
        print(f"           [ERROR] Failed: {e}")
        return None

def extract_excel_from_zip(zip_content):
    """Extract Excel files from ZIP"""
    excel_files = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            file_list = z.namelist()
            print(f"           [INFO] Found {len(file_list)} files in ZIP")

            for filename in file_list:
                if filename.endswith(('.xlsx', '.xls', '.csv')):
                    print(f"           [EXTRACT] {filename}")
                    file_data = z.read(filename)
                    excel_files.append({
                        "filename": filename,
                        "data": file_data
                    })

        print(f"           [SUCCESS] Extracted {len(excel_files)} data files")
        return excel_files

    except Exception as e:
        print(f"           [ERROR] Extraction failed: {e}")
        return []

def upload_csv_to_backend(df, dataset_name):
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
            'description': f'Baltimore community health data - {dataset_name}'
        }

        response = requests.post(
            f"{BACKEND_URL}/upload_csv",
            files=files,
            data=data,
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            print(f"         [SUCCESS] Ingested as UID: {result.get('uid', 'N/A')}")
            return True, result.get('uid')
        else:
            print(f"         [ERROR] Status {response.status_code}: {response.text[:200]}")
            return False, None

    except Exception as e:
        print(f"         [ERROR] Upload failed: {e}")
        return False, None

def main():
    print("="*80)
    print("DOWNLOAD AND INGEST BALTIMORE DATASETS")
    print("="*80)
    print("\nSources:")
    print("  - BNIA Vital Signs (2000-2010 historical data)")
    print("  - SAMHSA Baltimore-Towson reports")
    print("  - Additional Baltimore health datasets")
    print("\n" + "="*80)

    downloaded_datasets = []
    ingested_datasets = []

    # Step 1: Download datasets
    print("\n[STEP 1] DOWNLOADING BALTIMORE DATASETS")
    print("="*80)

    for source in BALTIMORE_DATA_SOURCES:
        content = download_file(source["url"], source["name"])

        if content:
            downloaded_datasets.append({
                "name": source["name"],
                "type": source["type"],
                "content": content,
                "description": source["description"]
            })

        time.sleep(2)  # Rate limiting

    print(f"\n[INFO] Successfully downloaded {len(downloaded_datasets)} datasets")

    # Step 2: Extract and process data files
    print("\n[STEP 2] PROCESSING DOWNLOADED DATA")
    print("="*80)

    for dataset in downloaded_datasets:
        if dataset["type"] == "zip":
            excel_files = extract_excel_from_zip(dataset["content"])

            for excel_file in excel_files:
                try:
                    # Try to read Excel/CSV
                    if excel_file["filename"].endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(excel_file["data"]))
                    else:
                        df = pd.read_excel(io.BytesIO(excel_file["data"]))

                    if len(df) > 0:
                        # Upload to backend
                        dataset_name = f"BNIA - {Path(excel_file['filename']).stem}"
                        success, uid = upload_csv_to_backend(df, dataset_name)

                        if success:
                            ingested_datasets.append({
                                "name": dataset_name,
                                "uid": uid,
                                "rows": len(df)
                            })

                        time.sleep(2)  # Rate limiting

                except Exception as e:
                    print(f"         [SKIP] Could not read {excel_file['filename']}: {e}")

        elif dataset["type"] == "pdf":
            print(f"\n[SKIP] {dataset['name']} - PDF format (not automatically parseable)")
            print(f"       Recommendation: Manual review for data extraction")

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD AND INGESTION COMPLETE")
    print("="*80)
    print(f"\n[SUMMARY]")
    print(f"  Datasets downloaded: {len(downloaded_datasets)}")
    print(f"  Datasets ingested: {len(ingested_datasets)}")

    if len(ingested_datasets) > 0:
        print(f"\n[INGESTED DATASETS]")
        for ds in ingested_datasets:
            print(f"  - {ds['name']}")
            print(f"    UID: {ds['uid']} | Rows: {ds['rows']:,}")

    print("\n[NEXT STEPS]")
    print("  1. Index the ingested datasets for Baltimore data")
    print("  2. Run: python index_all_baltimore.py")
    print("  3. Search for Baltimore health indicators in the UI")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
