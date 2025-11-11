"""
List all available datasets to help select 10 for export
"""
import requests
import json

response = requests.get("http://localhost:8000/datasets?org=All")
data = response.json()
datasets = data.get("items", [])

print(f"Total datasets available: {len(datasets)}\n")
print("=" * 100)

# Group by organization
orgs = {}
for d in datasets:
    org = d.get("organization", "Unknown")
    if org not in orgs:
        orgs[org] = []
    orgs[org].append(d)

for org, ds_list in orgs.items():
    print(f"\n{org} ({len(ds_list)} datasets):")
    print("-" * 100)
    for i, d in enumerate(ds_list[:10]):  # Show first 10 per org
        name = d.get("name", "")[:70]
        uid = d.get("uid", "")
        print(f"  {i+1}. {name}")
        print(f"     UID: {uid}")
        if "baltimore" in name.lower():
            print(f"     *** BALTIMORE DATA ***")
