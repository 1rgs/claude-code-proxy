#!/usr/bin/env python3
"""
Mini-test script to check if an Anthropic API key works.
"""

import httpx
import sys
import asyncio
import json

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

async def test_key(api_key):
    """Test if an Anthropic API key works."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    
    request_data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Say hello world"}
        ]
    }
    
    print(f"\nTesting Anthropic API key: {api_key[:10]}...{api_key[-6:]}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                json=request_data,
                headers=headers,
                timeout=10
            )
            
        if response.status_code == 200:
            response_json = response.json()
            print(f"✅ SUCCESS - API key works!")
            print(f"Response: {json.dumps(response_json, indent=2)[:200]}...")
            return True
        else:
            print(f"❌ FAILED - Status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

async def main():
    """Test multiple API keys."""
    if len(sys.argv) < 2:
        print("Usage: python test_anthropic_key.py API_KEY1 [API_KEY2 ...]")
        return 1
    
    keys = sys.argv[1:]
    results = []
    
    for key in keys:
        result = await test_key(key)
        results.append(result)
    
    if any(results):
        print("\n✅ At least one API key works!")
        for i, result in enumerate(results):
            if result:
                print(f"Working key: {keys[i][:10]}...{keys[i][-6:]}")
    else:
        print("\n❌ None of the API keys work.")
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())