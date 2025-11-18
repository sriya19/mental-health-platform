"""
Index all non-indexed datasets in the system (auto-confirm version)
"""
import requests
import time
from datetime import datetime

BACKEND_URL = "http://localhost:8000"

def get_all_datasets():
    """Get all datasets from backend"""
    response = requests.get(f"{BACKEND_URL}/datasets?org=All")
    if response.status_code == 200:
        return response.json().get("items", [])
    return []

def index_dataset(org, uid, dataset_name):
    """Index a single dataset"""
    print(f"\n{'='*80}")
    print(f"Indexing: {dataset_name}")
    print(f"Org: {org} | UID: {uid}")
    print(f"{'='*80}")

    try:
        response = requests.post(
            f"{BACKEND_URL}/index_dataset",
            json={
                "org": org,
                "uid": uid,
                "limit_rows": 20000
            },
            timeout=300  # 5 minute timeout for large datasets
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                chunks = result.get("chunks_created", 0)
                rows = result.get("rows_processed", 0)
                print(f"[SUCCESS] Indexed {chunks} chunks from {rows} rows")
                return True, chunks
            else:
                error = result.get("error", "Unknown error")
                print(f"[FAILED] {error}")
                return False, 0
        else:
            print(f"[HTTP ERROR] {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False, 0

    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] Indexing took longer than 5 minutes")
        return False, 0
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False, 0

def main():
    print("="*80)
    print("INDEXING ALL NON-INDEXED DATASETS (AUTO-CONFIRM)")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Wait for backend to be ready
    print("\nWaiting for backend to be ready...")
    time.sleep(3)

    # Get all datasets
    print("Fetching datasets from backend...")
    datasets = get_all_datasets()

    if not datasets:
        print("❌ No datasets found!")
        return

    print(f"Found {len(datasets)} total datasets")

    # Filter non-indexed datasets
    non_indexed = [d for d in datasets if not d.get("indexed_for_rag", False)]

    print(f"Found {len(non_indexed)} non-indexed datasets")

    if not non_indexed:
        print("[OK] All datasets are already indexed!")
        return

    # Show datasets to be indexed
    print("\nDatasets to index:")
    for i, d in enumerate(non_indexed, 1):
        uid = d.get("uid", "Unknown")
        org = d.get("org", "Unknown")
        name = d.get("name", "Unknown")
        print(f"  {i}. [{org}] {name[:50]} (UID: {uid})")

    # Auto-confirm
    print(f"\n{'='*80}")
    print(f"AUTO-CONFIRMING: Indexing all {len(non_indexed)} datasets...")
    print(f"{'='*80}")

    # Index each dataset
    successful = 0
    failed = 0
    total_chunks = 0

    for i, dataset in enumerate(non_indexed, 1):
        uid = dataset.get("uid")
        org = dataset.get("org", "CDC")
        name = dataset.get("name", "Unknown")

        # Skip if no UID
        if not uid or uid == "None":
            print(f"\n[{i}/{len(non_indexed)}] [WARNING] SKIPPING: {name} (No UID)")
            failed += 1
            continue

        print(f"\n[{i}/{len(non_indexed)}]")
        success, chunks = index_dataset(org, uid, name)

        if success:
            successful += 1
            total_chunks += chunks
        else:
            failed += 1

        # Small delay to avoid overwhelming the system
        if i < len(non_indexed):
            print(f"\nWaiting 2 seconds before next dataset...")
            time.sleep(2)

    # Summary
    print(f"\n{'='*80}")
    print("INDEXING COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
