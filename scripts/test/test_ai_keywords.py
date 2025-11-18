"""
Test AI-powered keyword extraction
"""
import requests
import json

BACKEND_URL = "http://localhost:8000"

# Test with your drug overdose query
test_queries = [
    "What are drug overdose trends in Baltimore?",
    "I need Baltimore mental health information",
    "Show me data on youth depression and suicide rates",
    "Cancer screening rates in urban areas",
    "COVID-19 vaccination data by state"
]

print("=" * 80)
print("TESTING AI-POWERED KEYWORD EXTRACTION")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n[Test {i}]")
    print(f"Query: {query}")
    print("-" * 80)

    response = requests.post(
        f"{BACKEND_URL}/semantic/search",
        json={
            "story": query,
            "org": "All",
            "k": 5,
            "persona": "Public health researcher"
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Found {len(result.get('results', []))} results")
        print(f"Sources: {result.get('sources', {})}")

        # Show top 3 results
        for j, r in enumerate(result.get('results', [])[:3], 1):
            name = r.get('name', 'Unknown')[:60]
            source = "ONLINE" if r.get('source') == 'socrata' else "LOCAL"
            print(f"  {j}. [{source}] {name}")
    else:
        print(f"Error: {response.status_code}")

print("\n" + "=" * 80)
print("Check backend logs to see AI-extracted keywords!")
print("Run: docker-compose logs backend --tail 50 | grep 'AI-extracted'")
print("=" * 80)
