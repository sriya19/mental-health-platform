import requests
import json

# Test unified search
response = requests.post(
    "http://localhost:8000/semantic/search",
    json={
        "story": "mental health",
        "org": "All",
        "k": 10
    }
)

print("Status:", response.status_code)
print("\nResponse:")
result = response.json()
print(json.dumps(result, indent=2))
print(f"\nTotal results: {len(result.get('results', []))}")
print(f"Sources: {result.get('sources', {})}")

# Show first few results
print("\nFirst 3 results:")
for i, r in enumerate(result.get('results', [])[:3], 1):
    print(f"\n{i}. {r.get('name', 'Unknown')}")
    print(f"   Org: {r.get('org', 'N/A')} | Source: {r.get('source', 'N/A')} | UID: {r.get('uid', 'N/A')}")
