"""
Test improved Baltimore search with location keywords
"""
import requests
import json

# Test the Baltimore search with improved keyword extraction
response = requests.post(
    "http://localhost:8000/semantic/search",
    json={
        "story": "I need Baltimore mental health information",
        "org": "All",
        "k": 10,
        "persona": "Public health researcher"
    }
)

print("=" * 80)
print("TESTING IMPROVED BALTIMORE SEARCH")
print("=" * 80)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()

    print(f"\nTotal results: {len(result.get('results', []))}")
    print(f"Sources breakdown: {result.get('sources', {})}")
    print(f"Model used: {result.get('model', 'N/A')}")

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    for i, r in enumerate(result.get('results', []), 1):
        source_badge = "[ONLINE]" if r.get('source') == 'socrata' else "[LOCAL]"
        print(f"\n{i}. {source_badge} {r.get('name', 'Unknown')}")
        print(f"   Org: {r.get('org', 'N/A')} | UID: {r.get('uid', 'N/A')}")

        # Show description preview
        desc = r.get('description', '')
        if desc:
            desc_preview = desc[:150].replace('\n', ' ')
            print(f"   Desc: {desc_preview}...")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Online (CDC/SAMHSA): {result.get('sources', {}).get('socrata', 0)}")
    print(f"Local (Uploaded CSVs): {result.get('sources', {}).get('local', 0)}")
    print("=" * 80)

    # The backend logs should show extracted keywords including 'baltimore'
    print("\nCheck backend logs to see extracted keywords (should include 'baltimore', 'mental', 'health')")

else:
    print(f"Error: {response.text}")
