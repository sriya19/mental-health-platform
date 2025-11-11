"""
Index ALL existing datasets for Baltimore/Maryland data only
This will filter and create Baltimore-specific chunks from all ingested datasets
"""
import requests
import time

BACKEND_URL = "http://localhost:8000"

def get_all_datasets():
    """Get list of all ingested datasets"""
    response = requests.get(f"{BACKEND_URL}/datasets?org=All")
    data = response.json()
    return data.get("items", [])

def index_baltimore_data(org, uid, name):
    """Index Baltimore/Maryland data only from a dataset"""
    try:
        print(f"\n[INDEX] {name}")
        print(f"        Org: {org} | UID: {uid}")

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
                total_rows = result.get("total_rows", 0)
                balt_rows = result.get("baltimore_rows", 0)
                chunks = result.get("chunks_created", 0)

                print(f"        [SUCCESS] Baltimore Data Found!")
                print(f"        Total rows: {total_rows:,}")
                print(f"        Baltimore rows: {balt_rows:,}")
                print(f"        Chunks created: {chunks}")
                return True, chunks
            else:
                reason = result.get("reason", "unknown")
                if reason == "no_baltimore_data":
                    print(f"        [SKIP] No Baltimore/Maryland data in this dataset")
                else:
                    print(f"        [SKIP] {reason}")
                return False, 0
        else:
            print(f"        [ERROR] Status {response.status_code}")
            return False, 0

    except Exception as e:
        print(f"        [ERROR] Failed: {e}")
        return False, 0

def main():
    print("="*80)
    print("INDEX ALL DATASETS FOR BALTIMORE DATA")
    print("="*80)
    print("\nThis script will:")
    print("  1. Get all ingested datasets")
    print("  2. Filter each for Baltimore/Maryland data")
    print("  3. Create Baltimore-specific RAG chunks")
    print("  4. Replace existing chunks with Baltimore-only chunks")
    print("\n" + "="*80)

    # Get all datasets
    print("\n[STEP 1] Fetching all ingested datasets...")
    datasets = get_all_datasets()
    print(f"[INFO] Found {len(datasets)} ingested datasets")

    # Index Baltimore data from each
    print("\n[STEP 2] Indexing Baltimore data from each dataset...")
    print("="*80)

    baltimore_datasets_count = 0
    total_chunks = 0
    processed = 0

    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}]")

        org = dataset.get("org", "Unknown")
        uid = dataset.get("uid", "")
        name = dataset.get("name", "Unknown Dataset")

        if not uid or uid == "None":
            print(f"[SKIP] {name} - No valid UID")
            continue

        success, chunks = index_baltimore_data(org, uid, name)

        if success:
            baltimore_datasets_count += 1
            total_chunks += chunks

        processed += 1
        time.sleep(1)  # Rate limiting

    # Summary
    print("\n" + "="*80)
    print("BALTIMORE DATA INDEXING COMPLETE")
    print("="*80)
    print(f"\n[SUMMARY]")
    print(f"  Total datasets processed: {processed}")
    print(f"  Datasets with Baltimore data: {baltimore_datasets_count}")
    print(f"  Total Baltimore chunks created: {total_chunks}")

    if baltimore_datasets_count > 0:
        print(f"\n[SUCCESS] Found Baltimore data in {baltimore_datasets_count} datasets!")
        print(f"[SUCCESS] Created {total_chunks} searchable Baltimore-specific chunks")
    else:
        print(f"\n[INFO] No datasets contained Baltimore/Maryland data")
        print(f"[INFO] This is normal if datasets are national-level only")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
