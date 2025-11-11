"""
Ingest New Maryland/Baltimore Datasets Found in Catalog
Automatically ingest and index datasets that contain Maryland data
"""
import requests
import time

BACKEND_URL = "http://localhost:8000"

# New Maryland datasets found in catalog (not yet ingested)
NEW_MARYLAND_DATASETS = [
    {"org": "CDC", "uid": "6tkz-y37d", "name": "NCHS - Birth Rates for Unmarried Women (Maryland data)"},
    {"org": "CDC", "uid": "g6qk-ngsf", "name": "NCHS - Births to Unmarried Women (Maryland data)"},
    {"org": "CDC", "uid": "hzd8-r9mj", "name": "NCHS - Percent Distribution Births Unmarried Women (Maryland data)"},
    {"org": "CDC", "uid": "tpcp-uiv5", "name": "Provisional COVID-19 Deaths by HHS Region, Race, Age (Maryland data)"},
    {"org": "CDC", "uid": "9xc7-3a4q", "name": "AH Provisional COVID-19 Deaths 65+ (Maryland data)"},
    {"org": "CDC", "uid": "k5dc-apj8", "name": "AH Provisional COVID-19 Deaths by HHS Region 2015-date (Maryland data)"},
    {"org": "CDC", "uid": "mpx5-t7tu", "name": "Provisional COVID-19 death counts by jurisdiction (Maryland data)"},
    {"org": "CDC", "uid": "yrur-wghw", "name": "Provisional COVID-19 death counts by demographics (Maryland data)"},
    {"org": "CDC", "uid": "w9j2-ggv5", "name": "NCHS - Death rates and life expectancy (Maryland data)"},
    {"org": "CDC", "uid": "mc4y-cbbv", "name": "NCHS - Top Five Leading Causes of Death (Maryland data)"},
    {"org": "CDC", "uid": "bi63-dtpu", "name": "NCHS - Leading Causes of Death (Maryland data)"},
    {"org": "CDC", "uid": "6rkc-nb2q", "name": "NCHS - Age-adjusted Death Rates Major Causes (Maryland data)"},
    {"org": "CDC", "uid": "v6ab-adf5", "name": "NCHS - Childhood Mortality Rates (Maryland data)"},
    {"org": "CDC", "uid": "nt65-c7a7", "name": "NCHS - Injury Mortality (Maryland data)"},
    {"org": "CDC", "uid": "89yk-m38d", "name": "NCHS - Natality Measures by Race (Maryland data)"},
    {"org": "CDC", "uid": "dmnu-8erf", "name": "Provisional COVID-19 deaths by demographics (Maryland data)"}
]

def ingest_dataset(org, uid, name):
    """Ingest a single dataset"""
    try:
        print(f"\n[INGEST] {name}")
        print(f"         Org: {org} | UID: {uid}")

        response = requests.post(
            f"{BACKEND_URL}/ingest",
            json={
                "org": org,
                "pick_uid": uid,
                "auto_index": False
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            rows = result.get('rows_ingested', 0)
            print(f"         [SUCCESS] Ingested {rows:,} rows")
            return True, rows
        else:
            error_msg = response.text[:200] if response.text else "No error message"
            print(f"         [ERROR] Status {response.status_code}: {error_msg}")
            return False, 0

    except Exception as e:
        print(f"         [ERROR] Failed: {e}")
        return False, 0

def index_baltimore_data(org, uid, name):
    """Index Baltimore/Maryland data only"""
    try:
        print(f"[INDEX] {name}")

        response = requests.post(
            f"{BACKEND_URL}/index_baltimore",
            json={
                "org": org,
                "uid": uid,
                "limit_rows": 20000
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()

            if result.get("success"):
                chunks = result.get("chunks_created", 0)
                balt_rows = result.get("baltimore_rows", 0)
                total_rows = result.get("total_rows", 0)

                print(f"        [SUCCESS] Found {balt_rows:,}/{total_rows:,} Baltimore/Maryland rows")
                print(f"        Created {chunks} searchable chunks")
                return True, chunks, balt_rows
            else:
                reason = result.get("reason", "Unknown")
                print(f"        [SKIP] {reason}")
                return False, 0, 0
        else:
            print(f"        [ERROR] Status {response.status_code}")
            return False, 0, 0

    except Exception as e:
        print(f"        [ERROR] Failed: {e}")
        return False, 0, 0

def main():
    print("="*80)
    print("INGEST NEW MARYLAND/BALTIMORE DATASETS")
    print("="*80)
    print(f"\nDatasets to ingest: {len(NEW_MARYLAND_DATASETS)}")
    print("These datasets contain Maryland/Baltimore data within national datasets")
    print("\n" + "="*80)

    ingested = []
    indexed = []
    total_chunks = 0
    total_baltimore_rows = 0

    # Step 1: Ingest datasets
    print("\n[STEP 1] INGESTING DATASETS")
    print("="*80)

    for i, dataset in enumerate(NEW_MARYLAND_DATASETS, 1):
        print(f"\n[{i}/{len(NEW_MARYLAND_DATASETS)}]")

        success, rows = ingest_dataset(
            dataset["org"],
            dataset["uid"],
            dataset["name"]
        )

        if success:
            ingested.append({
                "org": dataset["org"],
                "uid": dataset["uid"],
                "name": dataset["name"],
                "rows": rows
            })

        time.sleep(2)  # Rate limiting

    print(f"\n[INFO] Successfully ingested {len(ingested)}/{len(NEW_MARYLAND_DATASETS)} datasets")

    # Step 2: Index Baltimore/Maryland data
    if len(ingested) > 0:
        print("\n[STEP 2] INDEXING BALTIMORE/MARYLAND DATA")
        print("="*80)

        for i, dataset in enumerate(ingested, 1):
            print(f"\n[{i}/{len(ingested)}]")

            success, chunks, balt_rows = index_baltimore_data(
                dataset["org"],
                dataset["uid"],
                dataset["name"]
            )

            if success:
                indexed.append({
                    "name": dataset["name"],
                    "chunks": chunks,
                    "baltimore_rows": balt_rows
                })
                total_chunks += chunks
                total_baltimore_rows += balt_rows

            time.sleep(2)  # Rate limiting

    # Summary
    print("\n" + "="*80)
    print("NEW MARYLAND DATASET INGESTION COMPLETE")
    print("="*80)
    print(f"\n[SUMMARY]")
    print(f"  Datasets ingested: {len(ingested)}/{len(NEW_MARYLAND_DATASETS)}")
    print(f"  Datasets with Baltimore/Maryland data: {len(indexed)}")
    print(f"  Total Baltimore/Maryland rows: {total_baltimore_rows:,}")
    print(f"  Total new chunks: {total_chunks}")

    if len(indexed) > 0:
        print(f"\n[NEW BALTIMORE DATA INDEXED]")
        for ds in indexed[:10]:  # Show first 10
            print(f"  - {ds['name']}")
            print(f"    Baltimore rows: {ds['baltimore_rows']:,} | Chunks: {ds['chunks']}")

        if len(indexed) > 10:
            print(f"  ... and {len(indexed) - 10} more datasets")

        print(f"\n[SUCCESS] {total_chunks} new Baltimore chunks ready for search!")
        print(f"[SUCCESS] Platform now has comprehensive Maryland health data coverage")

    print("\n[OVERALL STATUS]")
    print(f"  Previous Baltimore chunks: 620")
    print(f"  New Baltimore chunks: {total_chunks}")
    print(f"  TOTAL Baltimore chunks: {620 + total_chunks}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
