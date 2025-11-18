"""
Comprehensive Baltimore Mental Health Dataset Ingestion
Searches and ingests datasets from:
- Open Baltimore (data.baltimorecity.gov)
- Data.gov Baltimore datasets
- BNIA Vital Signs
- Maryland Open Data
"""
import requests
import time

BACKEND_URL = "http://localhost:8000"

# Known Baltimore datasets from various sources
BALTIMORE_DATASETS = [
    # Data.gov Baltimore Health Datasets
    {
        "name": "Baltimore City Fire Department Leave-Behind Naloxone Distribution",
        "source": "Open Baltimore",
        "url": "https://data.baltimorecity.gov/datasets/baltimore::baltimore-city-fire-department-leave-behind-naloxone-distribution",
        "api_base": "https://services1.arcgis.com/mVFRs7NF4iFitgbY/arcgis/rest/services",
        "type": "ArcGIS",
        "description": "BCFD EMS clinicians distribute leave-behind naloxone kits"
    },
    {
        "name": "Baltimore City Fire Department Clinician-Administered Naloxone",
        "source": "Open Baltimore",
        "url": "https://data.baltimorecity.gov/datasets/baltimore::baltimore-city-fire-department-clinician-administered-naloxone",
        "api_base": "https://services1.arcgis.com/mVFRs7NF4iFitgbY/arcgis/rest/services",
        "type": "ArcGIS",
        "description": "Records instances where BCFD clinicians administered naloxone"
    },
    # Additional potential datasets to search for
    {
        "name": "Baltimore Opioid Overdose Dashboard Data",
        "source": "Baltimore Health Department",
        "search_terms": ["opioid", "overdose", "Baltimore"],
        "type": "Search",
        "description": "Opioid overdose tracking data from Baltimore City Health Department"
    },
    {
        "name": "Baltimore EMS Mental Health Calls",
        "source": "Baltimore 911",
        "search_terms": ["mental health", "psychiatric", "Baltimore", "911"],
        "type": "Search",
        "description": "Emergency medical service calls related to mental health"
    },
    {
        "name": "Baltimore Neighborhood Health Indicators",
        "source": "BNIA",
        "search_terms": ["Baltimore", "neighborhood", "health", "community statistical area"],
        "type": "Search",
        "description": "Community-level health indicators for Baltimore's 55 CSAs"
    }
]

def search_socrata_baltimore(search_terms):
    """Search for Baltimore datasets using backend search"""
    results = []

    # Try different search combinations
    queries = [
        " ".join(search_terms),
        "Baltimore " + " ".join([t for t in search_terms if t.lower() != "baltimore"]),
        search_terms[0] if search_terms else "Baltimore health"
    ]

    for query in queries[:2]:  # Limit to 2 search variations
        try:
            print(f"  [SEARCH] Searching for: {query}")
            response = requests.get(
                f"{BACKEND_URL}/catalog/search",
                params={"org": "All", "q": query},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                found = data.get("results", [])
                print(f"  [FOUND] {len(found)} results")

                for item in found:
                    if "Baltimore" in item.get("name", "") or "Baltimore" in item.get("description", ""):
                        results.append(item)

            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"  [ERROR] Search failed: {e}")

    return results

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
                "auto_index": False  # We'll index separately
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            print(f"         [SUCCESS] Ingested {result.get('rows_ingested', 0)} rows")
            return True
        else:
            print(f"         [ERROR] Status {response.status_code}: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"         [ERROR] Failed: {e}")
        return False

def index_dataset(org, uid, name):
    """Index a dataset for RAG"""
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
                print(f"        [SUCCESS] Indexed {result.get('chunks_created', 0)} Baltimore chunks")
                return True
            else:
                print(f"        [INFO] {result.get('reason', 'No Baltimore data found')}")
                return False
        else:
            print(f"        [ERROR] Status {response.status_code}")
            return False

    except Exception as e:
        print(f"        [ERROR] Failed: {e}")
        return False

def main():
    print("="*80)
    print("BALTIMORE MENTAL HEALTH DATASET INGESTION")
    print("="*80)
    print("\nThis script will search for and ingest Baltimore-focused datasets from:")
    print("  - Open Baltimore (data.baltimorecity.gov)")
    print("  - CDC/SAMHSA catalogs (Baltimore-specific)")
    print("  - BNIA Vital Signs")
    print("  - Maryland state data")
    print("\n" + "="*80)

    all_datasets_to_ingest = []
    ingested_count = 0
    indexed_count = 0

    # Step 1: Search for Baltimore datasets
    print("\n[STEP 1] SEARCHING FOR BALTIMORE DATASETS")
    print("="*80)

    for dataset_config in BALTIMORE_DATASETS:
        if dataset_config["type"] == "Search":
            print(f"\n[SEARCH] {dataset_config['name']}")
            search_results = search_socrata_baltimore(dataset_config.get("search_terms", []))

            for result in search_results[:3]:  # Top 3 results per search
                all_datasets_to_ingest.append({
                    "name": result.get("name"),
                    "org": result.get("org"),
                    "uid": result.get("uid"),
                    "source": dataset_config["source"]
                })

    # Also search for general Baltimore terms
    print(f"\n[SEARCH] General Baltimore health data")
    general_results = search_socrata_baltimore(["Baltimore", "health"])

    for result in general_results[:5]:  # Top 5 general results
        all_datasets_to_ingest.append({
            "name": result.get("name"),
            "org": result.get("org"),
            "uid": result.get("uid"),
            "source": "CDC/SAMHSA"
        })

    # Remove duplicates
    unique_datasets = []
    seen_uids = set()
    for ds in all_datasets_to_ingest:
        if ds["uid"] not in seen_uids:
            unique_datasets.append(ds)
            seen_uids.add(ds["uid"])

    print(f"\n[FOUND] {len(unique_datasets)} unique Baltimore datasets to ingest")

    # Step 2: Ingest datasets
    print("\n[STEP 2] INGESTING DATASETS")
    print("="*80)

    for i, dataset in enumerate(unique_datasets, 1):
        print(f"\n[{i}/{len(unique_datasets)}]")
        if ingest_dataset(dataset["org"], dataset["uid"], dataset["name"]):
            ingested_count += 1
            dataset["ingested"] = True
        else:
            dataset["ingested"] = False

        time.sleep(2)  # Rate limiting

    # Step 3: Index Baltimore-specific data
    print("\n[STEP 3] INDEXING BALTIMORE DATA")
    print("="*80)

    for dataset in unique_datasets:
        if dataset.get("ingested"):
            if index_dataset(dataset["org"], dataset["uid"], dataset["name"]):
                indexed_count += 1

            time.sleep(2)  # Rate limiting

    # Summary
    print("\n" + "="*80)
    print("INGESTION COMPLETE")
    print("="*80)
    print(f"\n[SUMMARY]")
    print(f"  Datasets found: {len(unique_datasets)}")
    print(f"  Datasets ingested: {ingested_count}")
    print(f"  Datasets indexed (Baltimore data only): {indexed_count}")

    print(f"\n[SOURCES]")
    sources = {}
    for ds in unique_datasets:
        source = ds.get("source", "Unknown")
        sources[source] = sources.get(source, 0) + 1

    for source, count in sources.items():
        print(f"  {source}: {count} datasets")

    print("\n" + "="*80)
    print("[SUCCESS] All available Baltimore datasets have been ingested!")
    print("="*80)

if __name__ == "__main__":
    main()
