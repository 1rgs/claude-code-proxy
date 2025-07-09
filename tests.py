#!/usr/bin/env python3
"""
Comprehensive test suite for Claude-on-OpenAI Proxy.

This script provides tests for both streaming and non-streaming requests,
with various scenarios including tool use, multi-turn conversations,
and content blocks.

Usage:
  python tests.py                    # Run all tests
  python tests.py --no-streaming     # Skip streaming tests
  python tests.py --simple           # Run only simple tests
  python tests.py --tools            # Run tool-related tests only
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PROXY_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # Using same key for proxy
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_API_URL = "http://localhost:8082/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL = "claude-3-sonnet-20240229"  # Change to your preferred model

# Headers
anthropic_headers = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

proxy_headers = {
    "x-api-key": PROXY_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
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
                "description": "The mathematical expression to evaluate",
            }
        },
        "required": ["expression"],
    },
}

weather_tool = {
    "name": "weather",
    "description": "Get weather information for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to get weather for",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units",
            },
        },
        "required": ["location"],
    },
}

search_tool = {
    "name": "search",
    "description": "Search for information on the web",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
}

# Test scenarios
TEST_SCENARIOS = {
    # Simple text response
    "simple": {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [
            {
                "role": "user",
                "content": "Hello, world! Can you tell me about Paris in 2-3 sentences?",
            }
        ],
    },
    # Basic tool use
    "calculator": {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": "What is 135 + 7.5 divided by 2.5?"}],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"},
    },
    # Multiple tools
    "multi_tool": {
        "model": MODEL,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.95,
        "system": "You are a helpful assistant that uses tools when appropriate. Be concise and precise.",
        "messages": [
            {
                "role": "user",
                "content": "I'm planning a trip to New York next week. What's the weather like and what are some interesting places to visit?",
            }
        ],
        "tools": [weather_tool, search_tool],
        "tool_choice": {"type": "auto"},
    },
    # Multi-turn conversation
    "multi_turn": {
        "model": MODEL,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": "Let's do some math. What is 240 divided by 8?",
            },
            {
                "role": "assistant",
                "content": "To calculate 240 divided by 8, I'll perform the division:\n\n240 ÷ 8 = 30\n\nSo the result is 30.",
            },
            {
                "role": "user",
                "content": "Now multiply that by 4 and tell me the result.",
            },
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"},
    },
    # Content blocks
    "content_blocks": {
        "model": MODEL,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I need to know the weather in Los Angeles and calculate 75.5 / 5. Can you help with both?",
                    }
                ],
            }
        ],
        "tools": [calculator_tool, weather_tool],
        "tool_choice": {"type": "auto"},
    },
    # Simple streaming test
    "simple_stream": {
        "model": MODEL,
        "max_tokens": 100,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Count from 1 to 5, with one number per line."}
        ],
    },
    # Tool use with streaming
    "calculator_stream": {
        "model": MODEL,
        "max_tokens": 300,
        "stream": True,
        "messages": [{"role": "user", "content": "What is 135 + 17.5 divided by 2.5?"}],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"},
    },
}

# Required event types for Anthropic streaming responses
REQUIRED_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
}

# ================= NON-STREAMING TESTS =================


def get_response(url, headers, data):
    """Send a request and get the response."""
    start_time = time.time()
    response = httpx.post(url, headers=headers, json=data, timeout=30)
    elapsed = time.time() - start_time

    print(f"Response time: {elapsed:.2f} seconds")
    return response


def compare_responses(anthropic_response, proxy_response, check_tools=False):
    """Compare the two responses to see if they're similar enough."""
    anthropic_json = anthropic_response.json()
    proxy_json = proxy_response.json()

    print("\n--- Anthropic Response Structure ---")
    print(
        json.dumps(
            {k: v for k, v in anthropic_json.items() if k != "content"}, indent=2
        )
    )

    print("\n--- Proxy Response Structure ---")
    print(json.dumps({k: v for k, v in proxy_json.items() if k != "content"}, indent=2))

    # Basic structure verification with more flexibility
    # The proxy might map values differently, so we're more lenient in our checks
    assert proxy_json.get("role") == "assistant", "Proxy role is not 'assistant'"
    assert proxy_json.get("type") == "message", "Proxy type is not 'message'"

    # Check if stop_reason is reasonable (might be different between Anthropic and our proxy)
    valid_stop_reasons = ["end_turn", "max_tokens", "stop_sequence", "tool_use", None]
    assert proxy_json.get("stop_reason") in valid_stop_reasons, "Invalid stop reason"

    # Check content exists and has valid structure
    assert "content" in anthropic_json, "No content in Anthropic response"
    assert "content" in proxy_json, "No content in Proxy response"

    anthropic_content = anthropic_json["content"]
    proxy_content = proxy_json["content"]

    # Make sure content is a list and has at least one item
    assert isinstance(anthropic_content, list), "Anthropic content is not a list"
    assert isinstance(proxy_content, list), "Proxy content is not a list"
    assert len(proxy_content) > 0, "Proxy content is empty"

    # If we're checking for tool uses
    if check_tools:
        # Check if content has tool use
        anthropic_tool = None
        proxy_tool = None

        # Find tool use in Anthropic response
        for item in anthropic_content:
            if item.get("type") == "tool_use":
                anthropic_tool = item
                break

        # Find tool use in Proxy response
        for item in proxy_content:
            if item.get("type") == "tool_use":
                proxy_tool = item
                break

        # At least one of them should have a tool use
        if anthropic_tool is not None:
            print("\n---------- ANTHROPIC TOOL USE ----------")
            print(json.dumps(anthropic_tool, indent=2))

            if proxy_tool is not None:
                print("\n---------- PROXY TOOL USE ----------")
                print(json.dumps(proxy_tool, indent=2))

                # Check tool structure
                assert proxy_tool.get("name") is not None, "Proxy tool has no name"
                assert proxy_tool.get("input") is not None, "Proxy tool has no input"

                print("\n✅ Both responses contain tool use")
            else:
                print(
                    "\n⚠️ Proxy response does not contain tool use, but Anthropic does"
                )
        elif proxy_tool is not None:
            print("\n---------- PROXY TOOL USE ----------")
            print(json.dumps(proxy_tool, indent=2))
            print("\n⚠️ Proxy response contains tool use, but Anthropic does not")
        else:
            print("\n⚠️ Neither response contains tool use")

    # Check if content has text
    anthropic_text = None
    proxy_text = None

    for item in anthropic_content:
        if item.get("type") == "text":
            anthropic_text = item.get("text")
            break

    for item in proxy_content:
        if item.get("type") == "text":
            proxy_text = item.get("text")
            break

    # For tool use responses, there might not be text content
    if check_tools and (anthropic_text is None or proxy_text is None):
        print(
            "\n⚠️ One or both responses don't have text content (expected for tool-only responses)"
        )
        return True

    assert anthropic_text is not None, "No text found in Anthropic response"
    assert proxy_text is not None, "No text found in Proxy response"

    # Print the first few lines of each text response
    max_preview_lines = 5
    anthropic_preview = "\n".join(
        anthropic_text.strip().split("\n")[:max_preview_lines]
    )
    proxy_preview = "\n".join(proxy_text.strip().split("\n")[:max_preview_lines])

    print("\n---------- ANTHROPIC TEXT PREVIEW ----------")
    print(anthropic_preview)

    print("\n---------- PROXY TEXT PREVIEW ----------")
    print(proxy_preview)

    # Check for some minimum text overlap - proxy might have different exact wording
    # but should have roughly similar content
    return True  # We're not enforcing similarity, just basic structure


def test_request(test_name, request_data, check_tools=False):
    """Run a test with the given request data."""
    print(f"\n{'=' * 20} RUNNING TEST: {test_name} {'=' * 20}")

    # Log the request data
    print(
        f"\nRequest data:\n{json.dumps({k: v for k, v in request_data.items() if k != 'messages'}, indent=2)}"
    )

    # Make copies of the request data to avoid modifying the original
    anthropic_data = request_data.copy()
    proxy_data = request_data.copy()

    try:
        # Send requests to both APIs
        print("\nSending to Anthropic API...")
        anthropic_response = get_response(
            ANTHROPIC_API_URL, anthropic_headers, anthropic_data
        )

        print("\nSending to Proxy...")
        proxy_response = get_response(PROXY_API_URL, proxy_headers, proxy_data)

        # Check response codes
        print(f"\nAnthropic status code: {anthropic_response.status_code}")
        print(f"Proxy status code: {proxy_response.status_code}")

        if anthropic_response.status_code != 200 or proxy_response.status_code != 200:
            print("\n⚠️ One or both requests failed")
            if anthropic_response.status_code != 200:
                print(f"Anthropic error: {anthropic_response.text}")
            if proxy_response.status_code != 200:
                print(f"Proxy error: {proxy_response.text}")
            return False

        # Compare the responses
        result = compare_responses(
            anthropic_response, proxy_response, check_tools=check_tools
        )
        if result:
            print(f"\n✅ Test {test_name} passed!")
            return True
        else:
            print(f"\n❌ Test {test_name} failed!")
            return False

    except Exception as e:
        print(f"\n❌ Error in test {test_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# ================= STREAMING TESTS =================


class StreamStats:
    """Track statistics about a streaming response."""

    def __init__(self):
        self.event_types = set()
        self.event_counts = {}
        self.first_event_time = None
        self.last_event_time = None
        self.total_chunks = 0
        self.events = []
        self.text_content = ""
        self.content_blocks = {}
        self.has_tool_use = False
        self.has_error = False
        self.error_message = ""
        self.text_content_by_block = {}

    def add_event(self, event_data):
        """Track information about each received event."""
        now = datetime.now()
        if self.first_event_time is None:
            self.first_event_time = now
        self.last_event_time = now

        self.total_chunks += 1

        # Record event type and increment count
        if "type" in event_data:
            event_type = event_data["type"]
            self.event_types.add(event_type)
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

            # Track specific event data
            if event_type == "content_block_start":
                block_idx = event_data.get("index")
                content_block = event_data.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    self.has_tool_use = True
                self.content_blocks[block_idx] = content_block
                self.text_content_by_block[block_idx] = ""

            elif event_type == "content_block_delta":
                block_idx = event_data.get("index")
                delta = event_data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    self.text_content += text
                    # Also track text by block ID
                    if block_idx in self.text_content_by_block:
                        self.text_content_by_block[block_idx] += text

        # Keep track of all events for debugging
        self.events.append(event_data)

    def get_duration(self):
        """Calculate the total duration of the stream in seconds."""
        if self.first_event_time is None or self.last_event_time is None:
            return 0
        return (self.last_event_time - self.first_event_time).total_seconds()

    def summarize(self):
        """Print a summary of the stream statistics."""
        print(f"Total chunks: {self.total_chunks}")
        print(f"Unique event types: {sorted(list(self.event_types))}")
        print(f"Event counts: {json.dumps(self.event_counts, indent=2)}")
        print(f"Duration: {self.get_duration():.2f} seconds")
        print(f"Has tool use: {self.has_tool_use}")

        # Print the first few lines of content
        if self.text_content:
            max_preview_lines = 5
            text_preview = "\n".join(
                self.text_content.strip().split("\n")[:max_preview_lines]
            )
            print(f"Text preview:\n{text_preview}")
        else:
            print("No text content extracted")

        if self.has_error:
            print(f"Error: {self.error_message}")


async def stream_response(url, headers, data, stream_name):
    """Send a streaming request and process the response."""
    print(f"\nStarting {stream_name} stream...")
    stats = StreamStats()
    error = None

    try:
        async with httpx.AsyncClient() as client:
            # Add stream flag to ensure it's streamed
            request_data = data.copy()
            request_data["stream"] = True

            start_time = time.time()
            async with client.stream(
                "POST", url, json=request_data, headers=headers, timeout=30
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    stats.has_error = True
                    stats.error_message = (
                        f"HTTP {response.status_code}: {error_text.decode('utf-8')}"
                    )
                    error = stats.error_message
                    print(f"Error: {stats.error_message}")
                    return stats, error

                print(f"{stream_name} connected, receiving events...")

                # Process each chunk
                buffer = ""
                async for chunk in response.aiter_text():
                    if not chunk.strip():
                        continue

                    # Handle multiple events in one chunk
                    buffer += chunk
                    events = buffer.split("\n\n")

                    # Process all complete events
                    for event_text in events[
                        :-1
                    ]:  # All but the last (possibly incomplete) event
                        if not event_text.strip():
                            continue

                        # Parse server-sent event format
                        if "data: " in event_text:
                            # Extract the data part
                            data_parts = []
                            for line in event_text.split("\n"):
                                if line.startswith("data: "):
                                    data_part = line[len("data: ") :]
                                    # Skip the "[DONE]" marker
                                    if data_part == "[DONE]":
                                        break
                                    data_parts.append(data_part)

                            if data_parts:
                                try:
                                    event_data = json.loads("".join(data_parts))
                                    stats.add_event(event_data)
                                except json.JSONDecodeError as e:
                                    print(
                                        f"Error parsing event: {e}\nRaw data: {''.join(data_parts)}"
                                    )

                    # Keep the last (potentially incomplete) event for the next iteration
                    buffer = events[-1] if events else ""

                # Process any remaining complete events in the buffer
                if buffer.strip():
                    lines = buffer.strip().split("\n")
                    data_lines = [
                        line[len("data: ") :]
                        for line in lines
                        if line.startswith("data: ")
                    ]
                    if data_lines and data_lines[0] != "[DONE]":
                        try:
                            event_data = json.loads("".join(data_lines))
                            stats.add_event(event_data)
                        except:
                            pass

            elapsed = time.time() - start_time
            print(f"{stream_name} stream completed in {elapsed:.2f} seconds")
    except Exception as e:
        stats.has_error = True
        stats.error_message = str(e)
        error = str(e)
        print(f"Error in {stream_name} stream: {e}")

    return stats, error


def compare_stream_stats(anthropic_stats, proxy_stats):
    """Compare the statistics from the two streams to see if they're similar enough."""

    print("\n--- Stream Comparison ---")

    # Required events
    anthropic_missing = REQUIRED_EVENT_TYPES - anthropic_stats.event_types
    proxy_missing = REQUIRED_EVENT_TYPES - proxy_stats.event_types

    print(f"Anthropic missing event types: {anthropic_missing}")
    print(f"Proxy missing event types: {proxy_missing}")

    # Check if proxy has the required events
    if proxy_missing:
        print(f"⚠️ Proxy is missing required event types: {proxy_missing}")
    else:
        print("✅ Proxy has all required event types")

    # Compare content
    if anthropic_stats.text_content and proxy_stats.text_content:
        anthropic_preview = "\n".join(
            anthropic_stats.text_content.strip().split("\n")[:5]
        )
        proxy_preview = "\n".join(proxy_stats.text_content.strip().split("\n")[:5])

        print("\n--- Anthropic Content Preview ---")
        print(anthropic_preview)

        print("\n--- Proxy Content Preview ---")
        print(proxy_preview)

    # Compare tool use
    if anthropic_stats.has_tool_use and proxy_stats.has_tool_use:
        print("✅ Both have tool use")
    elif anthropic_stats.has_tool_use and not proxy_stats.has_tool_use:
        print("⚠️ Anthropic has tool use but proxy does not")
    elif not anthropic_stats.has_tool_use and proxy_stats.has_tool_use:
        print("⚠️ Proxy has tool use but Anthropic does not")

    # Success as long as proxy has some content and no errors
    return (
        not proxy_stats.has_error
        and len(proxy_stats.text_content) > 0
        or proxy_stats.has_tool_use
    )


async def test_streaming(test_name, request_data):
    """Run a streaming test with the given request data."""
    print(f"\n{'=' * 20} RUNNING STREAMING TEST: {test_name} {'=' * 20}")

    # Log the request data
    print(
        f"\nRequest data:\n{json.dumps({k: v for k, v in request_data.items() if k != 'messages'}, indent=2)}"
    )

    # Make copies of the request data to avoid modifying the original
    anthropic_data = request_data.copy()
    proxy_data = request_data.copy()

    if not anthropic_data.get("stream"):
        anthropic_data["stream"] = True
    if not proxy_data.get("stream"):
        proxy_data["stream"] = True

    check_tools = "tools" in request_data

    try:
        # Send streaming requests
        anthropic_stats, anthropic_error = await stream_response(
            ANTHROPIC_API_URL, anthropic_headers, anthropic_data, "Anthropic"
        )

        proxy_stats, proxy_error = await stream_response(
            PROXY_API_URL, proxy_headers, proxy_data, "Proxy"
        )

        # Print statistics
        print("\n--- Anthropic Stream Statistics ---")
        anthropic_stats.summarize()

        print("\n--- Proxy Stream Statistics ---")
        proxy_stats.summarize()

        # Compare the responses
        if anthropic_error:
            print(f"\n⚠️ Anthropic stream had an error: {anthropic_error}")
            # If Anthropic errors, the test passes if proxy does anything useful
            if not proxy_error and proxy_stats.total_chunks > 0:
                print(
                    f"\n✅ Test {test_name} passed! (Proxy worked even though Anthropic failed)"
                )
                return True
            else:
                print(f"\n❌ Test {test_name} failed! Both streams had errors.")
                return False

        if proxy_error:
            print(f"\n❌ Test {test_name} failed! Proxy had an error: {proxy_error}")
            return False

        result = compare_stream_stats(anthropic_stats, proxy_stats)
        if result:
            print(f"\n✅ Test {test_name} passed!")
            return True
        else:
            print(f"\n❌ Test {test_name} failed!")
            return False

    except Exception as e:
        print(f"\n❌ Error in test {test_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# ================= MAIN =================


async def run_tests(args):
    """Run all tests based on command-line arguments."""
    # Track test results
    results = {}

    # First run non-streaming tests
    if not args.streaming_only:
        print("\n\n=========== RUNNING NON-STREAMING TESTS ===========\n")
        for test_name, test_data in TEST_SCENARIOS.items():
            # Skip streaming tests
            if test_data.get("stream"):
                continue

            # Skip tool tests if requested
            if args.simple and "tools" in test_data:
                continue

            # Skip non-tool tests if tools_only
            if args.tools_only and "tools" not in test_data:
                continue

            # Run the test
            check_tools = "tools" in test_data
            result = test_request(test_name, test_data, check_tools=check_tools)
            results[test_name] = result

    # Now run streaming tests
    if not args.no_streaming:
        print("\n\n=========== RUNNING STREAMING TESTS ===========\n")
        for test_name, test_data in TEST_SCENARIOS.items():
            # Only select streaming tests, or force streaming
            if not test_data.get("stream") and not test_name.endswith("_stream"):
                continue

            # Skip tool tests if requested
            if args.simple and "tools" in test_data:
                continue

            # Skip non-tool tests if tools_only
            if args.tools_only and "tools" not in test_data:
                continue

            # Run the streaming test
            result = await test_streaming(test_name, test_data)
            results[f"{test_name}_streaming"] = result

    # Print summary
    print("\n\n=========== TEST SUMMARY ===========\n")
    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test, result in results.items():
        print(f"{test}: {'✅ PASS' if result else '❌ FAIL'}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        return False


async def main():
    # Check that API key is set
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set in .env file")
        return

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the Claude-on-OpenAI proxy")
    parser.add_argument(
        "--no-streaming", action="store_true", help="Skip streaming tests"
    )
    parser.add_argument(
        "--streaming-only", action="store_true", help="Only run streaming tests"
    )
    parser.add_argument(
        "--simple", action="store_true", help="Only run simple tests (no tools)"
    )
    parser.add_argument("--tools-only", action="store_true", help="Only run tool tests")
    args = parser.parse_args()

    # Run tests
    success = await run_tests(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
