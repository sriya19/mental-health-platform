"""
Check current indexing status
"""
import requests

BACKEND_URL = "http://localhost:8000"

response = requests.get(f"{BACKEND_URL}/datasets?org=All")

if response.status_code == 200:
    data = response.json()
    datasets = data.get("items", [])

    total = len(datasets)
    indexed = len([d for d in datasets if d.get("indexed_for_rag", False)])
    not_indexed = total - indexed
    total_chunks = sum(d.get("chunk_count", 0) for d in datasets)

    print("=" * 80)
    print("INDEXING STATUS")
    print("=" * 80)
    print(f"Total Datasets:     {total}")
    print(f"Indexed:            {indexed}")
    print(f"Not Indexed:        {not_indexed}")
    print(f"Total Chunks:       {total_chunks}")
    print("=" * 80)

    if not_indexed > 0:
        print("\nNot Indexed Datasets:")
        for d in datasets:
            if not d.get("indexed_for_rag", False):
                name = d.get("name", "Unknown")[:50]
                org = d.get("org", "Unknown")
                uid = d.get("uid", "None")
                print(f"  - [{org}] {name} (UID: {uid})")
else:
    print(f"Error: {response.status_code}")
