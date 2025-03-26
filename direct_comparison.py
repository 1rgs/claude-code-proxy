#!/usr/bin/env python3
"""
Direct comparison between Anthropic API and our proxy.
"""

import httpx
import time
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
PROXY_URL = "http://localhost:8082/v1/messages"

# Test data
HAIKU_REQUEST = {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 300,
    "messages": [
        {"role": "user", "content": "Hello, what is Paris known for? Please answer in 2-3 sentences."}
    ]
}

SONNET_REQUEST = {
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1000,
    "messages": [
        {"role": "user", "content": "If a train travels at 120 km/h and another train travels at 180 km/h in the opposite direction, and they start 900 km apart, how long will it take for them to meet?"}
    ]
}

CODE_REQUEST = {
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1000,
    "messages": [
        {"role": "user", "content": "Write a Python function to check if a string is a palindrome."}
    ]
}

def test_anthropic(request):
    """Test Anthropic API."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    start_time = time.time()
    response = httpx.post(
        ANTHROPIC_URL,
        json=request,
        headers=headers,
        timeout=30
    )
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        response_json = response.json()
        return {
            "success": True,
            "time": elapsed,
            "response": response_json
        }
    else:
        return {
            "success": False,
            "time": elapsed,
            "error": response.text
        }

def test_proxy(request):
    """Test proxy API."""
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    start_time = time.time()
    response = httpx.post(
        PROXY_URL,
        json=request,
        headers=headers,
        timeout=60
    )
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        response_json = response.json()
        return {
            "success": True,
            "time": elapsed,
            "response": response_json
        }
    else:
        return {
            "success": False,
            "time": elapsed,
            "error": response.text
        }

def run_comparison(name, request):
    """Run comparison test."""
    print(f"\n=== Testing {name} ===")
    
    # Test Anthropic
    print("Testing Anthropic API...")
    anthropic_result = test_anthropic(request)
    if anthropic_result["success"]:
        print(f"✅ Success - {anthropic_result['time']:.2f}s")
        # Get text from first content block
        text = ""
        if "content" in anthropic_result["response"]:
            for block in anthropic_result["response"]["content"]:
                if block.get("type") == "text":
                    text = block.get("text", "")[:100] + "..."
                    break
        print(f"Response: {text}")
    else:
        print(f"❌ Failed - {anthropic_result['time']:.2f}s")
        print(f"Error: {anthropic_result['error']}")
    
    # Test Proxy
    print("\nTesting Proxy...")
    proxy_result = test_proxy(request)
    if proxy_result["success"]:
        print(f"✅ Success - {proxy_result['time']:.2f}s")
        # Get text from first content block
        text = ""
        if "content" in proxy_result["response"]:
            for block in proxy_result["response"]["content"]:
                if block.get("type") == "text":
                    text = block.get("text", "")[:100] + "..."
                    break
        print(f"Response: {text}")
    else:
        print(f"❌ Failed - {proxy_result['time']:.2f}s")
        print(f"Error: {proxy_result['error']}")
    
    # Compare
    if anthropic_result["success"] and proxy_result["success"]:
        ratio = proxy_result["time"] / anthropic_result["time"]
        diff = proxy_result["time"] - anthropic_result["time"]
        print(f"\nComparison: Proxy is {ratio:.2f}x slower ({diff:+.2f}s)")
    
    return {
        "name": name,
        "anthropic": anthropic_result,
        "proxy": proxy_result
    }

def main():
    """Run comparisons."""
    # Check server is running
    try:
        response = httpx.get(PROXY_URL.replace("/messages", ""), timeout=2)
        print("Proxy server is running.")
    except:
        print("Error: Proxy server is not running. Start it first.")
        return 1
    
    # Check API key
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set.")
        return 1
    
    # Run tests
    results = []
    
    results.append(run_comparison("Simple text (Haiku)", HAIKU_REQUEST))
    results.append(run_comparison("Complex reasoning (Sonnet)", SONNET_REQUEST))
    results.append(run_comparison("Code generation (Sonnet)", CODE_REQUEST))
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'Test':<25} {'Anthropic':<12} {'Proxy':<12} {'Ratio':<8} {'Diff'}")
    print("-" * 70)
    
    for result in results:
        name = result["name"]
        
        if result["anthropic"]["success"]:
            anthropic_time = f"{result['anthropic']['time']:.2f}s"
        else:
            anthropic_time = "Failed"
            
        if result["proxy"]["success"]:
            proxy_time = f"{result['proxy']['time']:.2f}s"
        else:
            proxy_time = "Failed"
            
        if result["anthropic"]["success"] and result["proxy"]["success"]:
            ratio = result["proxy"]["time"] / result["anthropic"]["time"]
            diff = result["proxy"]["time"] - result["anthropic"]["time"]
            ratio_str = f"{ratio:.2f}x"
            diff_str = f"{diff:+.2f}s"
        else:
            ratio_str = "N/A"
            diff_str = "N/A"
            
        print(f"{name:<25} {anthropic_time:<12} {proxy_time:<12} {ratio_str:<8} {diff_str}")
    
    print("\nThis comparison shows the performance difference between Anthropic API and our proxy.")
    print("The 'Ratio' column shows how many times slower the proxy is compared to Anthropic.")
    print("The 'Diff' column shows the absolute time difference in seconds.")
    
    # Save results
    with open("direct_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to direct_comparison_results.json")
    
    return 0

if __name__ == "__main__":
    exit(main())