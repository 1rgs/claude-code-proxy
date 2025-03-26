#!/usr/bin/env python3
"""
Simple performance comparison script for Claude Code proxy.
"""

import os
import json
import time
import httpx
import asyncio
import statistics
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_API_URL = "http://localhost:8082/v1/messages"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Headers
anthropic_headers = {
    "x-api-key": ANTHROPIC_API_KEY if ANTHROPIC_API_KEY else "invalid-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

proxy_headers = {
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
}

# Test scenarios
TESTS = [
    {
        "name": "simple_text",
        "description": "Simple text generation",
        "model": "claude-3-haiku-20240307",
        "request": {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 300,
            "messages": [
                {"role": "user", "content": "Hello, what is Paris known for? Please answer in 2-3 sentences."}
            ]
        }
    },
    {
        "name": "complex_reasoning",
        "description": "Complex mathematical reasoning",
        "model": "claude-3-sonnet-20240229",
        "request": {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": "If a train travels at 120 km/h and another train travels at 180 km/h in the opposite direction, and they start 900 km apart, how long will it take for them to meet?"}
            ]
        }
    },
    {
        "name": "code_generation",
        "description": "Python code generation",
        "model": "claude-3-sonnet-20240229",
        "request": {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": "Write a Python function to check if a string is a palindrome."}
            ]
        }
    }
]

async def test_model(name, description, anthropic_req=None, proxy_req=None, iterations=3):
    """Run tests against both APIs and measure performance."""
    results = {
        "name": name,
        "description": description,
        "anthropic": {"times": [], "tokens": {"input": [], "output": []}},
        "proxy": {"times": [], "tokens": {"input": [], "output": []}},
        "timestamp": datetime.now().isoformat()
    }
    
    # Test Anthropic API if request provided
    if anthropic_req and ANTHROPIC_API_KEY:
        print(f"\n=== Testing Anthropic API: {name} ({iterations} iterations) ===")
        for i in range(iterations):
            try:
                start_time = time.time()
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        ANTHROPIC_API_URL, 
                        json=anthropic_req,
                        headers=anthropic_headers,
                        timeout=30
                    )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    response_json = response.json()
                    print(f"  Iteration {i+1}: ✅ Success - {elapsed:.2f}s")
                    
                    # Record metrics
                    results["anthropic"]["times"].append(elapsed)
                    if "usage" in response_json:
                        results["anthropic"]["tokens"]["input"].append(response_json["usage"]["input_tokens"])
                        results["anthropic"]["tokens"]["output"].append(response_json["usage"]["output_tokens"])
                else:
                    print(f"  Iteration {i+1}: ❌ Failed ({response.status_code}) - {elapsed:.2f}s")
                    print(f"  Error: {response.text}")
            except Exception as e:
                print(f"  Iteration {i+1}: ❌ Error - {str(e)}")
    
    # Test Proxy API if request provided
    if proxy_req:
        print(f"\n=== Testing Proxy API: {name} ({iterations} iterations) ===")
        for i in range(iterations):
            try:
                start_time = time.time()
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        PROXY_API_URL, 
                        json=proxy_req,
                        headers=proxy_headers,
                        timeout=60
                    )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    response_json = response.json()
                    print(f"  Iteration {i+1}: ✅ Success - {elapsed:.2f}s")
                    
                    # Record metrics
                    results["proxy"]["times"].append(elapsed)
                    if "usage" in response_json:
                        results["proxy"]["tokens"]["input"].append(response_json["usage"]["input_tokens"])
                        results["proxy"]["tokens"]["output"].append(response_json["usage"]["output_tokens"])
                else:
                    print(f"  Iteration {i+1}: ❌ Failed ({response.status_code}) - {elapsed:.2f}s")
                    print(f"  Error: {response.text}")
            except Exception as e:
                print(f"  Iteration {i+1}: ❌ Error - {str(e)}")
    
    # Calculate summary statistics
    if results["anthropic"]["times"]:
        results["anthropic"]["avg_time"] = sum(results["anthropic"]["times"]) / len(results["anthropic"]["times"])
        if len(results["anthropic"]["times"]) > 1:
            results["anthropic"]["std_dev"] = statistics.stdev(results["anthropic"]["times"])
    
    if results["proxy"]["times"]:
        results["proxy"]["avg_time"] = sum(results["proxy"]["times"]) / len(results["proxy"]["times"])
        if len(results["proxy"]["times"]) > 1:
            results["proxy"]["std_dev"] = statistics.stdev(results["proxy"]["times"])
    
    # Calculate comparison if both have data
    if results["anthropic"]["times"] and results["proxy"]["times"]:
        anthropic_avg = results["anthropic"]["avg_time"]
        proxy_avg = results["proxy"]["avg_time"]
        results["comparison"] = {
            "difference_seconds": proxy_avg - anthropic_avg,
            "ratio": proxy_avg / anthropic_avg if anthropic_avg > 0 else float('inf')
        }
    
    return results

async def run_tests(iterations=3, server_type="default"):
    """Run all tests and generate summary."""
    all_results = []
    
    for test in TESTS:
        # Use the same request data for both APIs
        results = await test_model(
            test["name"],
            test["description"],
            test["request"],
            test["request"],
            iterations
        )
        all_results.append(results)
    
    # Save results to file
    output_filename = f"performance_comparison_{server_type}.json"
    with open(output_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_filename}")
    
    # Print summary table
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Server configuration: {server_type}")
    print("----------------------------------------------------------")
    print(f"{'Test':<20} {'Anthropic':<12} {'Proxy':<12} {'Ratio':<8} {'Diff (s)'}")
    print("----------------------------------------------------------")
    
    for result in all_results:
        name = result["name"]
        
        if "avg_time" in result["anthropic"]:
            anthropic_time = f"{result['anthropic']['avg_time']:.2f}s"
        else:
            anthropic_time = "N/A"
            
        if "avg_time" in result["proxy"]:
            proxy_time = f"{result['proxy']['avg_time']:.2f}s"
        else:
            proxy_time = "N/A"
            
        if "comparison" in result:
            ratio = f"{result['comparison']['ratio']:.2f}x"
            diff = f"{result['comparison']['difference_seconds']:+.2f}"
        else:
            ratio = "N/A"
            diff = "N/A"
            
        print(f"{name:<20} {anthropic_time:<12} {proxy_time:<12} {ratio:<8} {diff}")
    
    print("----------------------------------------------------------")
    
    return all_results

async def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run performance comparison tests")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each test")
    parser.add_argument("--server-type", type=str, default="default", help="Server configuration label")
    parser.add_argument("--proxy-only", action="store_true", help="Only test proxy (skip Anthropic API)")
    args = parser.parse_args()
    
    # Check server is running
    try:
        async with httpx.AsyncClient() as client:
            await client.get(PROXY_API_URL.replace("/messages", ""), timeout=2)
    except:
        print("Error: Proxy server is not running. Start it with:")
        print("  python server.py [--always-cot]")
        return 1
    
    # Run tests
    await run_tests(args.iterations, args.server_type)
    return 0

if __name__ == "__main__":
    asyncio.run(main())