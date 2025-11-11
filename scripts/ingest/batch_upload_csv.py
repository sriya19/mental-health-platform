"""
Batch upload mental health CSV datasets to the platform
"""
import os
import requests
from pathlib import Path
import time

# Configuration
BACKEND_URL = "http://localhost:8000"
DOWNLOADS_DIR = r"C:\Users\sanka\Downloads"

# Define datasets to upload with metadata
DATASETS = [
    # BRFSS files (already uploaded one, but keeping for reference)
    {
        "filename": "Behavioral_Risk_Factor_Surveillance_System_(BRFSS)_-_Mental_Health_Indicators_20250831.csv",
        "name": "BRFSS Mental Health Indicators 2025-08-31",
        "description": "Behavioral Risk Factor Surveillance System - Mental Health Indicators",
        "org": "BRFSS"
    },
    {
        "filename": "Behavioral_Risk_Factor_Surveillance_System_(BRFSS)_-_Mental_Health_Indicators_20250904 (1).csv",
        "name": "BRFSS Mental Health Indicators 2025-09-04 (1)",
        "description": "Behavioral Risk Factor Surveillance System - Mental Health Indicators",
        "org": "BRFSS"
    },
    {
        "filename": "Behavioral_Risk_Factor_Surveillance_System_(BRFSS)_-_Mental_Health_Indicators_20250904.csv",
        "name": "BRFSS Mental Health Indicators 2025-09-04",
        "description": "Behavioral Risk Factor Surveillance System - Mental Health Indicators",
        "org": "BRFSS"
    },
    {
        "filename": "Behavioral_Risk_Factor_Surveillance_System_(BRFSS)_-_Mental_Health_Indicators_20251105.csv",
        "name": "BRFSS Mental Health Indicators 2025-11-05",
        "description": "Behavioral Risk Factor Surveillance System - Mental Health Indicators",
        "org": "BRFSS"
    },

    # NHIS files
    {
        "filename": "National_Health_Interview_Survey_(NHIS)_-_Mental_Health_Indicators_20250831.csv",
        "name": "NHIS Mental Health Indicators 2025-08-31",
        "description": "National Health Interview Survey - Mental Health Indicators",
        "org": "NHIS"
    },
    {
        "filename": "National_Health_Interview_Survey_(NHIS)_-_Mental_Health_Indicators_20250904.csv",
        "name": "NHIS Mental Health Indicators 2025-09-04",
        "description": "National Health Interview Survey - Mental Health Indicators",
        "org": "NHIS"
    },
    {
        "filename": "National_Health_Interview_Survey_(NHIS)_-_Mental_Health_Indicators_20251105.csv",
        "name": "NHIS Mental Health Indicators 2025-11-05",
        "description": "National Health Interview Survey - Mental Health Indicators",
        "org": "NHIS"
    },

    # YRBSS files
    {
        "filename": "Youth Risk Behavioral Surveillance System (YRBSS) - Mental Health Indicators_20251109_140146.csv",
        "name": "YRBSS Mental Health Indicators 2025-11-09",
        "description": "Youth Risk Behavioral Surveillance System - Mental Health Indicators",
        "org": "YRBSS"
    },
    {
        "filename": "Youth_Risk_Behavioral_Surveillance_System_(YRBSS)_-_Mental_Health_Indicators_20250904.csv",
        "name": "YRBSS Mental Health Indicators 2025-09-04",
        "description": "Youth Risk Behavioral Surveillance System - Mental Health Indicators",
        "org": "YRBSS"
    },
    {
        "filename": "Youth_Risk_Behavioral_Surveillance_System_(YRBSS)_-_Mental_Health_Indicators_20251107.csv",
        "name": "YRBSS Mental Health Indicators 2025-11-07",
        "description": "Youth Risk Behavioral Surveillance System - Mental Health Indicators",
        "org": "YRBSS"
    },

    # Other mental health datasets
    {
        "filename": "Mental_Health_Care_in_the_Last_4_Weeks_20251021.csv",
        "name": "Mental Health Care in Last 4 Weeks",
        "description": "Mental Health Care access and utilization in the last 4 weeks",
        "org": "CDC"
    },
    {
        "filename": "National_Syndromic_Surveillance_Program__NSSP__Mental_Health-Related_Emergency_Department_Visit_Rates.csv",
        "name": "NSSP Mental Health ED Visit Rates",
        "description": "National Syndromic Surveillance Program - Mental Health-Related Emergency Department Visit Rates",
        "org": "NSSP"
    },
    {
        "filename": "Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States.csv",
        "name": "Suicide Death Rates by Demographics",
        "description": "Death rates for suicide by sex, race, Hispanic origin, and age in the United States",
        "org": "CDC"
    },
    {
        "filename": "DASH___School_Health_Profiles__Profiles_.csv",
        "name": "DASH School Health Profiles",
        "description": "Division of Adolescent and School Health - School Health Profiles",
        "org": "CDC"
    },
    {
        "filename": "Indicators_of_Health_Insurance_Coverage_at_the_Time_of_Interview.csv",
        "name": "Health Insurance Coverage Indicators",
        "description": "Indicators of Health Insurance Coverage at the Time of Interview",
        "org": "CDC"
    }
]

def upload_csv(dataset_info):
    """Upload a single CSV file to the platform"""
    filepath = os.path.join(DOWNLOADS_DIR, dataset_info["filename"])

    # Check if file exists
    if not os.path.exists(filepath):
        print(f"[X] File not found: {dataset_info['filename']}")
        return False

    print(f"\n[UPLOADING] {dataset_info['name']}")
    print(f"   File: {dataset_info['filename']}")

    try:
        with open(filepath, 'rb') as f:
            files = {'file': (dataset_info['filename'], f, 'text/csv')}
            data = {
                'dataset_name': dataset_info['name'],
                'description': dataset_info['description'],
                'org': dataset_info['org'],
                'auto_index': 'true'  # Enable auto-indexing for RAG
            }

            response = requests.post(
                f"{BACKEND_URL}/upload_csv",
                files=files,
                data=data,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                print(f"   [SUCCESS]")
                print(f"      - Rows: {result.get('rows', 'N/A')}")
                print(f"      - Columns: {len(result.get('columns', []))}")
                print(f"      - Dataset UID: {result.get('dataset_uid', 'N/A')}")
                print(f"      - Indexed: {result.get('indexed', False)}")
                print(f"      - Chunks: {result.get('chunks_created', 0)}")
                return True
            else:
                print(f"   [FAILED] {response.status_code}")
                print(f"      {response.text}")
                return False

    except Exception as e:
        print(f"   [ERROR] {str(e)}")
        return False

def main():
    """Main function to upload all datasets"""
    print("=" * 70)
    print("Mental Health Platform - Batch CSV Upload")
    print("=" * 70)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Downloads Directory: {DOWNLOADS_DIR}")
    print(f"Total datasets to upload: {len(DATASETS)}")
    print("=" * 70)

    # Check backend health
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Backend is healthy and ready")
        else:
            print("[ERROR] Backend is not responding properly")
            return
    except Exception as e:
        print(f"[ERROR] Cannot connect to backend: {e}")
        return

    # Upload all datasets
    successful = 0
    failed = 0
    skipped = 0

    for i, dataset in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}]", end=" ")

        # Add a small delay between uploads to avoid overwhelming the server
        if i > 1:
            time.sleep(1)

        if upload_csv(dataset):
            successful += 1
        else:
            # Ask if we should skip the file that doesn't exist
            filepath = os.path.join(DOWNLOADS_DIR, dataset["filename"])
            if not os.path.exists(filepath):
                skipped += 1
            else:
                failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("UPLOAD SUMMARY")
    print("=" * 70)
    print(f"[SUCCESS] Successful: {successful}")
    print(f"[FAILED] Failed: {failed}")
    print(f"[SKIPPED] Skipped (file not found): {skipped}")
    print(f"[TOTAL] Total: {len(DATASETS)}")
    print("=" * 70)

    print("\n[COMPLETE] Batch upload complete!")
    print(f"\n[INFO] You can now query your data at: http://localhost:8501")
    print(f"[INFO] View API docs at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
