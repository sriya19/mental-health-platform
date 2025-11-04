#!/usr/bin/env python3
"""
Diagnostic script to check why RAG is failing
Run this to see what's wrong with your system
"""

import requests
import json

BACKEND_URL = "http://localhost:8000"

def check_system():
    print("=" * 60)
    print("MENTAL HEALTH PLATFORM - RAG DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Check backend health
    print("\n1. Checking backend health...")
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            print("✅ Backend is running")
        else:
            print(f"❌ Backend returned status {r.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("Make sure your backend is running on port 8000")
        return
    
    # 2. Check if semantic_index table has data
    print("\n2. Checking semantic_index table...")
    try:
        # Check CDC semantic index
        r = requests.post(
            f"{BACKEND_URL}/semantic/search",
            json={"story": "test", "org": "CDC", "k": 1}
        )
        data = r.json()
        
        if r.status_code == 200:
            if data.get("results"):
                print(f"✅ Semantic index has {len(data.get('results', []))} datasets")
            else:
                print("⚠️ Semantic index is empty - need to reindex")
                print("   Run: curl -X POST 'http://localhost:8000/semantic/reindex?org=CDC&limit=20'")
        else:
            print(f"❌ Semantic search failed: {data}")
    except Exception as e:
        print(f"❌ Semantic search error: {e}")
    
    # 3. Check if any datasets are ingested
    print("\n3. Checking ingested datasets...")
    try:
        r = requests.get(f"{BACKEND_URL}/datasets?org=CDC")
        data = r.json()
        count = data.get("count", 0)
        if count > 0:
            print(f"✅ Found {count} ingested CDC datasets")
            # Show first 3
            for item in data.get("items", [])[:3]:
                print(f"   - {item['name'][:50]}...")
        else:
            print("⚠️ No datasets ingested yet")
            print("   Use the 'Search & Results' tab to ingest some datasets first")
    except Exception as e:
        print(f"❌ Error checking datasets: {e}")
    
    # 4. Test if OpenAI API is configured
    print("\n4. Testing embedding capability...")
    try:
        # Try a simple semantic search which will use embeddings
        r = requests.post(
            f"{BACKEND_URL}/semantic/search",
            json={"story": "depression", "org": "CDC", "k": 1},
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("used_semantic") == False and data.get("fallback") == "keyword":
                print("⚠️ Embeddings not working - falling back to keyword search")
                print("   Check your OPENAI_API_KEY in .env file")
            elif data.get("used_semantic"):
                print("✅ Embeddings are working")
            else:
                print("❓ Unknown embedding status")
        else:
            print(f"❌ Embedding test failed: {r.status_code}")
    except requests.Timeout:
        print("❌ Embedding request timed out - check your OpenAI API key")
    except Exception as e:
        print(f"❌ Embedding error: {e}")
    
    # 5. Test the answer endpoint directly
    print("\n5. Testing answer generation...")
    try:
        r = requests.post(
            f"{BACKEND_URL}/answer",
            json={
                "question": "What is depression?",
                "org": "CDC",
                "k": 3,
                "persona": "researcher"
            },
            timeout=30
        )
        if r.status_code == 200:
            data = r.json()
            mode = data.get("mode", "unknown")
            if data.get("ok"):
                print(f"✅ Answer endpoint working (mode: {mode})")
            else:
                print(f"❌ Answer generation failed")
        else:
            print(f"❌ Answer endpoint returned {r.status_code}")
            print(f"   Response: {r.text[:200]}")
    except requests.Timeout:
        print("❌ Answer generation timed out (>30s)")
        print("   This is your main problem!")
    except Exception as e:
        print(f"❌ Answer endpoint error: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    print("\nMOST LIKELY ISSUES:")
    print("1. Semantic index is empty - run the reindex command")
    print("2. OpenAI API key is missing or invalid")
    print("3. Network/firewall blocking OpenAI API calls")
    print("4. Insufficient datasets ingested")

if __name__ == "__main__":
    check_system()