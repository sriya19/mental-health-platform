"""
Test unified search with a user story
"""
import requests
import json

# Test with a real user story
user_story = "As a public health researcher, I need data on youth depression and mental health trends to identify at-risk populations"

print("=" * 80)
print("TESTING UNIFIED SEARCH")
print("=" * 80)
print(f"User Story: {user_story}")
print(f"Organization: All")
print("=" * 80)

# Call the semantic search endpoint (same as Streamlit uses)
response = requests.post(
    "http://localhost:8000/semantic/search",
    json={
        "story": user_story,
        "org": "All",
        "k": 15,
        "persona": "Public health researcher"
    }
)

print(f"\nStatus Code: {response.status_code}")

if response.status_code == 200:
    result = response.json()

    print(f"\nResults Found: {len(result.get('results', []))}")
    print(f"Sources: {result.get('sources', {})}")
    print(f"Semantic Search Used: {result.get('used_semantic', False)}")
    print(f"Model: {result.get('model', 'N/A')}")

    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    for i, r in enumerate(result.get('results', []), 1):
        source_badge = "🌐 ONLINE" if r.get('source') == 'socrata' else "💾 LOCAL"
        print(f"\n[{i}] {source_badge}")
        print(f"    Name: {r.get('name', 'Unknown')}")
        print(f"    Org: {r.get('org', 'N/A')}")
        print(f"    UID: {r.get('uid', 'N/A')}")
        print(f"    Source: {r.get('source', 'N/A')}")

        desc = r.get('description', '')
        if desc:
            print(f"    Description: {desc[:100]}...")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Results: {len(result.get('results', []))}")

    online_count = len([r for r in result.get('results', []) if r.get('source') == 'socrata'])
    local_count = len([r for r in result.get('results', []) if r.get('source') == 'local'])

    print(f"Online (CDC/SAMHSA): {online_count}")
    print(f"Local (Your Uploads): {local_count}")
    print("=" * 80)

else:
    print(f"\nError: {response.text}")
