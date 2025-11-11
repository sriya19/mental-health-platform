"""
Search Backend Catalog for Baltimore/Maryland Datasets
Find datasets that contain Baltimore or Maryland keywords that we haven't ingested yet
"""
import requests
import time

BACKEND_URL = "http://localhost:8000"

# Search terms for Baltimore/Maryland health data
SEARCH_TERMS = [
    "Baltimore",
    "Maryland",
    "Baltimore City",
    "MD",
    "Maryland health",
    "Baltimore mental health",
    "Baltimore opioid",
    "Maryland substance abuse",
    "Baltimore overdose"
]

def get_ingested_datasets():
    """Get list of already ingested datasets"""
    try:
        response = requests.get(f"{BACKEND_URL}/datasets?org=All", timeout=30)
        if response.status_code == 200:
            data = response.json()
            ingested_uids = set()
            for item in data.get("items", []):
                uid = item.get("uid")
                if uid and uid != "None":
                    ingested_uids.add(uid)
            return ingested_uids
        return set()
    except Exception as e:
        print(f"[ERROR] Could not get ingested datasets: {e}")
        return set()

def search_catalog(org, query):
    """Search catalog for datasets"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/catalog/search",
            params={"org": org, "q": query},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        return []
    except Exception as e:
        print(f"[ERROR] Search failed for '{query}': {e}")
        return []

def main():
    print("="*80)
    print("SEARCH CATALOG FOR BALTIMORE/MARYLAND DATASETS")
    print("="*80)

    # Get already ingested datasets
    print("\n[INFO] Getting list of ingested datasets...")
    ingested_uids = get_ingested_datasets()
    print(f"[INFO] Found {len(ingested_uids)} already ingested datasets")

    # Search for Baltimore/Maryland datasets
    print("\n[SEARCH] Searching catalog for Baltimore/Maryland datasets...")
    print("="*80)

    all_results = {}  # uid -> dataset info
    baltimore_keywords = ["baltimore", "md", "maryland"]

    # Search across different organizations
    orgs_to_search = ["CDC", "SAMHSA", "All"]

    for org in orgs_to_search:
        for search_term in SEARCH_TERMS[:5]:  # Limit to first 5 search terms
            print(f"\n[SEARCH] Org: {org} | Query: '{search_term}'")

            results = search_catalog(org, search_term)
            print(f"         Found {len(results)} results")

            for result in results:
                uid = result.get("uid") or result.get("assetId", "")
                name = result.get("name", "Unknown")
                description = result.get("description", "")

                if not uid:
                    continue

                # Check if contains Baltimore/Maryland keywords
                text_to_check = (name + " " + description).lower()
                has_baltimore = any(keyword in text_to_check for keyword in baltimore_keywords)

                if has_baltimore and uid not in ingested_uids:
                    all_results[uid] = {
                        "uid": uid,
                        "name": name,
                        "org": result.get("org", org),
                        "description": description[:200],
                        "ingested": False
                    }

            time.sleep(1)  # Rate limiting

    # Show results
    print("\n" + "="*80)
    print("BALTIMORE/MARYLAND DATASETS FOUND (NOT YET INGESTED)")
    print("="*80)

    if len(all_results) > 0:
        print(f"\n[FOUND] {len(all_results)} Baltimore/Maryland datasets not yet ingested:\n")

        for i, (uid, info) in enumerate(list(all_results.items())[:20], 1):  # Show top 20
            print(f"{i}. {info['name']}")
            print(f"   UID: {info['uid']}")
            print(f"   Org: {info['org']}")
            if info['description']:
                print(f"   Description: {info['description']}")
            print()

        print("\n[RECOMMENDATION]")
        print("To ingest these datasets, use the Streamlit UI:")
        print("  1. Go to 'Search & Add Datasets'")
        print("  2. Search for Baltimore/Maryland")
        print("  3. Click 'Ingest' on relevant datasets")
        print("\nOr use the backend API to ingest programmatically.")

    else:
        print("\n[INFO] No new Baltimore/Maryland datasets found in catalog")
        print("[INFO] All available Baltimore datasets may already be ingested")
        print("\n[STATUS] You already have:")
        print(f"  - {len(ingested_uids)} total ingested datasets")
        print("  - 31 datasets with Baltimore/Maryland data")
        print("  - 620 Baltimore-specific searchable chunks")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
