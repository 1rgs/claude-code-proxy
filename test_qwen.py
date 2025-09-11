#!/usr/bin/env python3
"""
Simple test script to verify Qwen model configuration
Run this after setting up your .env file with Qwen/Dashscope credentials
"""

import os
import json
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PROXY_API_URL = "http://localhost:8082/v1/messages"
API_KEY = os.environ.get("OPENAI_API_KEY")  # Should be your Dashscope API key

if not API_KEY:
    print("❌ Error: OPENAI_API_KEY not set in .env file")
    print("Please copy .env.qwen.example to .env and add your Dashscope API key")
    exit(1)

headers = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

# Test cases for different Claude model names that should map to Qwen
test_cases = [
    {
        "name": "Claude Haiku (should map to SMALL_MODEL)",
        "data": {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello! Say 'Qwen model working' if you can see this."}
            ]
        }
    },
    {
        "name": "Claude Sonnet (should map to BIG_MODEL)", 
        "data": {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello! Say 'Qwen model working' if you can see this."}
            ]
        }
    },
    {
        "name": "Direct Qwen model",
        "data": {
            "model": "qwen3-coder-flash",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello! Say 'Qwen model working' if you can see this."}
            ]
        }
    }
]

def test_request(test_case):
    """Test a single request"""
    print(f"\n{'='*20} {test_case['name']} {'='*20}")
    print(f"Model: {test_case['data']['model']}")
    
    try:
        response = httpx.post(PROXY_API_URL, headers=headers, json=test_case['data'], timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Response Model: {result.get('model', 'Unknown')}")
            
            # Extract and print the text content
            content = result.get('content', [])
            for block in content:
                if block.get('type') == 'text':
                    print(f"Response: {block.get('text', '')[:200]}...")
                    break
        else:
            print(f"❌ Failed!")
            print(f"Error: {response.text}")
        
        return response.status_code == 200
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    print("🔧 Testing Qwen Model Configuration")
    print("Make sure the proxy server is running on http://localhost:8082")
    
    # Check if server is reachable
    try:
        response = httpx.get("http://localhost:8082/", timeout=5)
        if response.status_code != 200:
            print("❌ Proxy server not reachable. Start it with:")
            print("uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload")
            return
    except:
        print("❌ Proxy server not reachable. Start it with:")
        print("uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload")
        return
    
    print("✅ Proxy server is running")
    
    # Run tests
    results = []
    for test_case in test_cases:
        success = test_request(test_case)
        results.append(success)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_case, success) in enumerate(zip(test_cases, results)):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_case['name']}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Qwen model tests passed!")
        print("Your Qwen configuration is working correctly!")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("Check your .env configuration and Dashscope API key")
        print("Also verify that OPENAI_BASE_URL is set to: https://dashscope.aliyuncs.com/compatible-mode/v1")

if __name__ == "__main__":
    main()