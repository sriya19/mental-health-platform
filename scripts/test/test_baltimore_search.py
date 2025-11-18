import requests
import json

# Test the exact search you did
response = requests.post(
    "http://localhost:8000/semantic/search",
    json={
        "story": "I need Baltimore mental health information",
        "org": "All",
        "k": 10,
        "persona": "Public health researcher"
    }
)

print("Status:", response.status_code)
result = response.json()

print(f"\nTotal results: {len(result.get('results', []))}")
print(f"Sources breakdown: {result.get('sources', {})}")

print("\n" + "="*80)
print("ALL RESULTS:")
print("="*80)

for i, r in enumerate(result.get('results', []), 1):
    source_type = "ONLINE (CDC/SAMHSA)" if r.get('source') == 'socrata' else "LOCAL (Your Upload)"
    print(f"\n[{i}] {source_type}")
    print(f"    Name: {r.get('name')}")
    print(f"    Org: {r.get('org')}")
    print(f"    UID: {r.get('uid')}")
