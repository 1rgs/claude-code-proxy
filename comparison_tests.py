#!/usr/bin/env python3
"""
Comparison testing script for Claude-on-Deepseek/Gemini Proxy vs. Anthropic API.

This script runs identical tests against both the proxy and the direct Anthropic API,
measuring and comparing performance metrics including:
- Response time
- Response quality (token count)
- Success rate for different task types

Usage:
  python comparison_tests.py --anthropic-key=your-key [--proxy-only] [--anthropic-only]
  
Environment variables:
  ANTHROPIC_API_KEY: Your Anthropic API key (required for Anthropic API tests)
  DEEPSEEK_API_KEY: Your Deepseek API key (for proxy)
  GEMINI_API_KEY: Your Gemini API key (for proxy)
"""

import os
import json
import time
import httpx
import argparse
import asyncio
import sys
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from dotenv import load_dotenv
import tabulate
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# URLs
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_API_URL = "http://localhost:8082/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Headers
anthropic_headers = {
    "x-api-key": ANTHROPIC_API_KEY if ANTHROPIC_API_KEY else "invalid-key",
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

proxy_headers = {
    "content-type": "application/json",
    "anthropic-version": ANTHROPIC_VERSION,
}

# Tool definitions
calculator_tool = {
    "name": "calculator",
    "description": "Evaluate mathematical expressions",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}

# Test scenarios - represent different types of tasks
TEST_SCENARIOS = {
    # Simple text response
    "simple_text": {
        "description": "Simple text response (Haiku/Gemini)",
        "model": "claude-3-haiku-20240307",
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "Hello, what is Paris known for? Please answer in 2-3 sentences."}
        ]
    },
    
    # Tool usage
    "tool_usage": {
        "description": "Calculator tool usage (Haiku/Gemini)",
        "model": "claude-3-haiku-20240307",
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "Calculate 135 + 7.5 divided by 2.5?"}
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    },
    
    # Complex reasoning
    "complex_reasoning": {
        "description": "Complex reasoning (Sonnet/Deepseek)",
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": "If a train travels at 120 km/h and another train travels at 180 km/h in the opposite direction, and they start 900 km apart, how long will it take for them to meet?"}
        ]
    },
    
    # Code generation
    "code_generation": {
        "description": "Python code generation (Sonnet/Deepseek)",
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": "Write a Python function to check if a string is a palindrome."}
        ]
    }
}

class TestResult:
    """Store test results and metrics."""
    def __init__(self, test_name: str, api_type: str):
        self.test_name = test_name
        self.api_type = api_type
        self.status = False  # True for pass, False for fail
        self.response_time = 0.0
        self.token_count = {"input": 0, "output": 0}
        self.error = None
        self.http_status = None
        self.response_text = ""
        self.raw_response = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "test_name": self.test_name,
            "api_type": self.api_type,
            "status": "PASS" if self.status else "FAIL",
            "response_time_seconds": round(self.response_time, 2),
            "token_count": self.token_count,
            "error": str(self.error) if self.error else None,
            "http_status": self.http_status
        }

class ComparisonResults:
    """Store and analyze results from comparison tests."""
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        self.end_time = None
    
    def add_result(self, result: TestResult):
        """Add a test result to the collection."""
        self.results.append(result)
    
    def complete(self):
        """Mark completion of tests."""
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """Get total test duration in seconds."""
        if not self.end_time:
            self.end_time = datetime.now()
        return (self.end_time - self.start_time).total_seconds()
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary of test results."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status)
        
        # Group results by API type and test name
        anthropic_results = [r for r in self.results if r.api_type == "anthropic"]
        proxy_results = [r for r in self.results if r.api_type == "proxy"]
        
        # Calculate average response times by test type
        avg_times = {}
        comparison = {}
        
        for test_name in TEST_SCENARIOS.keys():
            anthropic_times = [r.response_time for r in anthropic_results if r.test_name == test_name and r.status]
            proxy_times = [r.response_time for r in proxy_results if r.test_name == test_name and r.status]
            
            if anthropic_times and proxy_times:
                avg_anthropic = sum(anthropic_times) / len(anthropic_times)
                avg_proxy = sum(proxy_times) / len(proxy_times)
                avg_times[test_name] = {
                    "anthropic": round(avg_anthropic, 2),
                    "proxy": round(avg_proxy, 2)
                }
                
                # Calculate the ratio (how many times slower/faster)
                ratio = avg_proxy / avg_anthropic if avg_anthropic > 0 else 0
                comparison[test_name] = {
                    "ratio": round(ratio, 2),
                    "difference_seconds": round(avg_proxy - avg_anthropic, 2)
                }
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            "duration_seconds": round(self.get_duration(), 2),
            "average_response_times": avg_times,
            "comparison": comparison
        }
    
    def generate_chart(self) -> BytesIO:
        """Generate a comparison chart of response times."""
        summary = self.get_summary()
        avg_times = summary["average_response_times"]
        
        # Extract data for the chart
        tests = list(avg_times.keys())
        anthropic_times = [avg_times[t]["anthropic"] for t in tests]
        proxy_times = [avg_times[t]["proxy"] for t in tests]
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set the width of the bars and positions
        width = 0.35
        x = range(len(tests))
        
        # Create the bars
        anthropic_bars = ax.bar([i - width/2 for i in x], anthropic_times, width, label='Anthropic API')
        proxy_bars = ax.bar([i + width/2 for i in x], proxy_times, width, label='Proxy (Deepseek/Gemini)')
        
        # Add labels, title, and legend
        ax.set_xlabel('Test Scenarios')
        ax.set_ylabel('Response Time (seconds)')
        ax.set_title('Response Time Comparison: Anthropic API vs. Proxy')
        ax.set_xticks(x)
        ax.set_xticklabels(tests, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels above the bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        add_labels(anthropic_bars)
        add_labels(proxy_bars)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save to BytesIO buffer
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        return buf

async def run_test(test_name: str, test_data: Dict[str, Any], api_type: str = "anthropic", iterations: int = 1) -> List[TestResult]:
    """Run a single test against the specified API."""
    results = []
    
    # Determine URL and headers
    if api_type == "anthropic":
        url = ANTHROPIC_API_URL
        headers = anthropic_headers
    else:  # proxy
        url = PROXY_API_URL
        headers = proxy_headers
    
    # Run the test for the specified number of iterations
    for i in range(iterations):
        result = TestResult(test_name, api_type)
        
        try:
            # Make a copy of the test data
            request_data = test_data.copy()
            
            # For Sonnet models on Proxy, extend timeout
            timeout_val = 60 if "sonnet" in request_data.get("model", "").lower() and api_type == "proxy" else 30
            
            # Send the request and measure response time
            start_time = time.time()
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=request_data, headers=headers, timeout=timeout_val)
            result.response_time = time.time() - start_time
            
            # Record HTTP status
            result.http_status = response.status_code
            
            # Process response
            if response.status_code == 200:
                response_json = response.json()
                result.raw_response = response_json
                
                # Extract token counts
                if "usage" in response_json:
                    usage = response_json["usage"]
                    result.token_count["input"] = usage.get("input_tokens", 0)
                    result.token_count["output"] = usage.get("output_tokens", 0)
                
                # Extract text content for analysis
                if "content" in response_json and isinstance(response_json["content"], list):
                    for item in response_json["content"]:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                result.response_text = text
                                break
                
                # Mark as successful
                result.status = True
            else:
                # Record failure
                result.status = False
                result.error = f"HTTP {response.status_code}: {response.text}"
        
        except Exception as e:
            # Record exception
            result.status = False
            result.error = str(e)
        
        results.append(result)
    
    return results

async def run_comparison_tests(test_iterations: int = 1, proxy_only: bool = False, anthropic_only: bool = False) -> ComparisonResults:
    """Run all tests against both APIs and compare results."""
    results = ComparisonResults()
    
    for test_name, test_data in TEST_SCENARIOS.items():
        print(f"\n=== Running test: {test_name} ({test_data['description']}) ===")
        
        # Run against Anthropic API (if not proxy_only)
        if not proxy_only and ANTHROPIC_API_KEY:
            print(f"  Running against Anthropic API ({test_iterations} iterations)...")
            test_results = await run_test(test_name, test_data, "anthropic", test_iterations)
            for result in test_results:
                results.add_result(result)
                print(f"    Iteration {test_results.index(result)+1}: {'✓ PASS' if result.status else '✗ FAIL'} - {result.response_time:.2f}s")
        
        # Run against Proxy (if not anthropic_only)
        if not anthropic_only:
            print(f"  Running against Proxy API ({test_iterations} iterations)...")
            test_results = await run_test(test_name, test_data, "proxy", test_iterations)
            for result in test_results:
                results.add_result(result)
                print(f"    Iteration {test_results.index(result)+1}: {'✓ PASS' if result.status else '✗ FAIL'} - {result.response_time:.2f}s")
    
    # Mark tests as complete
    results.complete()
    return results

def print_comparison_table(results: ComparisonResults):
    """Print a formatted comparison table."""
    summary = results.get_summary()
    comparison = summary["comparison"]
    avg_times = summary["average_response_times"]
    
    # Prepare table data
    table_data = []
    for test_name in TEST_SCENARIOS.keys():
        if test_name in avg_times:
            anthropic_time = avg_times[test_name]["anthropic"]
            proxy_time = avg_times[test_name]["proxy"]
            ratio = comparison[test_name]["ratio"]
            difference = comparison[test_name]["difference_seconds"]
            
            table_data.append([
                test_name,
                TEST_SCENARIOS[test_name]["description"],
                f"{anthropic_time:.2f}s",
                f"{proxy_time:.2f}s",
                f"{ratio:.2f}x",
                f"{difference:+.2f}s"
            ])
    
    # Print the table
    headers = ["Test", "Description", "Anthropic", "Proxy", "Ratio (Proxy/Anthropic)", "Difference"]
    print("\n=== Response Time Comparison ===")
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print summary
    print(f"\nTotal Duration: {summary['duration_seconds']:.2f} seconds")
    print(f"Success Rate: {summary['success_rate']}% ({summary['passed_tests']}/{summary['total_tests']} tests passed)")

def save_results_to_file(results: ComparisonResults, filename: str = "comparison_results.json", label: str = ""):
    """Save test results to a JSON file."""
    # Prepare data for serialization
    data = {
        "summary": results.get_summary(),
        "results": [r.to_dict() for r in results.results],
        "timestamp": datetime.now().isoformat()
    }
    
    # Apply label to filename if provided
    if label:
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{label}{ext}"
    
    # Write to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {filename}")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comparison tests between Anthropic API and Proxy")
    parser.add_argument("--anthropic-key", help="Anthropic API key")
    parser.add_argument("--proxy-only", action="store_true", help="Only test against the proxy")
    parser.add_argument("--anthropic-only", action="store_true", help="Only test against Anthropic API")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations for each test")
    parser.add_argument("--label", type=str, help="Label to add to output files")
    args = parser.parse_args()
    
    # Override API key if provided as argument
    global ANTHROPIC_API_KEY
    if args.anthropic_key:
        ANTHROPIC_API_KEY = args.anthropic_key
    
    # Check for API keys
    if not ANTHROPIC_API_KEY and not args.proxy_only:
        print("Error: Anthropic API key is required for tests. Set ANTHROPIC_API_KEY in .env or use --anthropic-key")
        print("Alternatively, use --proxy-only to skip Anthropic API tests")
        return 1
    
    if args.proxy_only and args.anthropic_only:
        print("Error: Cannot use both --proxy-only and --anthropic-only")
        return 1
    
    # Check if proxy server is running
    if not args.anthropic_only:
        try:
            async with httpx.AsyncClient() as client:
                await client.get(PROXY_API_URL.replace("/messages", ""), timeout=2)
        except:
            print("Error: Proxy server is not running. Please start the server before running tests.")
            print("Run: python server.py --always-cot")
            return 1
    
    # Run tests
    try:
        print(f"Starting comparison tests ({args.iterations} iterations per test)...")
        results = await run_comparison_tests(args.iterations, args.proxy_only, args.anthropic_only)
        
        # Print results
        print_comparison_table(results)
        
        # Save results with label if provided
        label = ""
        if args.label:
            label = args.label
        save_results_to_file(results, label=label)
        
        # Generate chart if both APIs were tested
        if not args.proxy_only and not args.anthropic_only:
            chart_file = "comparison_chart.png"
            if label:
                base, ext = os.path.splitext(chart_file)
                chart_file = f"{base}_{label}{ext}"
            with open(chart_file, "wb") as f:
                f.write(results.generate_chart().getvalue())
            print(f"Comparison chart saved to {chart_file}")
        
        return 0
    
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\nError running tests: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))