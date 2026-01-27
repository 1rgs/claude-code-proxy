"""Direct NVIDIA NIM provider using raw httpx - bypasses all libraries."""

import asyncio
import logging
import os
import json
import uuid
from typing import Dict, Any, AsyncIterator, Union, Optional
import time
import threading
from collections import deque
import httpx
from .base import BaseProvider, ProviderConfig

logger = logging.getLogger(__name__)


class NVIDIANIMDirectProvider(BaseProvider):
    """Direct NVIDIA NIM provider using raw httpx requests."""

    # Rate limiting
    _rate_lock = threading.Condition()
    _request_timestamps = deque()

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._api_key = config.api_key or os.getenv("NVIDIA_NIM_API_KEY", "")
        self._base_url = (
            config.base_url
            or os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        ).rstrip("/")
        self._nim_params = self._load_nim_params()
        # Initialize a persistent client for connection reuse
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    def _load_nim_params(self) -> Dict[str, Any]:
        """Load NIM parameters from environment variables."""
        params: Dict[str, Any] = {}
        if val := os.getenv("NVIDIA_NIM_TEMPERATURE"):
            params["temperature"] = float(val)
        if val := os.getenv("NVIDIA_NIM_TOP_P"):
            params["top_p"] = float(val)
        if val := os.getenv("NVIDIA_NIM_MAX_TOKENS"):
            params["max_tokens"] = int(val)
        return params

    def get_model_prefix(self) -> str:
        return ""

    async def _enforce_rate_limit_async(self):
        """Enforce rate limit using asyncio.sleep so the event loop is not blocked.

        This is the preferred method for async contexts (FastAPI, etc.).
        Uses non-blocking asyncio.sleep() with capped wait times for responsiveness.
        """
        rate_limit = self.config.rate_limit or 40
        rate_window = self.config.rate_window or 60
        max_retries = 120  # Prevent infinite loops (120 * 1s = 120s max wait)
        retry_count = 0

        while retry_count < max_retries:
            now = time.time()
            with self._rate_lock:
                # Clean up old timestamps outside the rate window
                while (
                    self._request_timestamps
                    and now - self._request_timestamps[0] > rate_window
                ):
                    self._request_timestamps.popleft()

                # Check if we can proceed
                if len(self._request_timestamps) < rate_limit:
                    self._request_timestamps.append(now)
                    logger.debug(
                        f"NVIDIA NIM Direct request recorded. Count: {len(self._request_timestamps)}/{rate_limit}"
                    )
                    return

                # Calculate wait time
                wait_time = self._request_timestamps[0] + rate_window - now

            if wait_time <= 0:
                # Timestamps should have been cleaned, retry immediately
                retry_count += 1
                continue

            # Wait using non-blocking asyncio.sleep (capped at 1 second to stay responsive)
            actual_wait = min(wait_time, 1.0)
            logger.debug(
                f"NVIDIA NIM Direct rate limit reached. Waiting {actual_wait:.2f}s (need {wait_time:.2f}s total)"
            )
            await asyncio.sleep(actual_wait)
            retry_count += 1

        # If we exceeded max retries, log warning but proceed anyway
        logger.warning(
            f"NVIDIA NIM Direct rate limit wait exceeded {max_retries} retries, proceeding anyway"
        )
        with self._rate_lock:
            self._request_timestamps.append(time.time())

    def _convert_messages(self, messages) -> list:
        """Convert Anthropic messages to OpenAI format."""
        result = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    text_parts = []
                    tool_calls = []
                    reasoning_parts = []

                    for block in content:
                        block_type = (
                            block.type
                            if hasattr(block, "type")
                            else block.get("type")
                            if isinstance(block, dict)
                            else None
                        )

                        if block_type == "text":
                            text = (
                                block.text
                                if hasattr(block, "text")
                                else block.get("text", "")
                            )
                            text_parts.append(text)
                        elif block_type == "thinking":
                            thinking = (
                                block.thinking
                                if hasattr(block, "thinking")
                                else block.get("thinking", "")
                            )
                            reasoning_parts.append(thinking)
                        elif block_type == "tool_use":
                            tool_id = (
                                block.id if hasattr(block, "id") else block.get("id")
                            )
                            tool_name = (
                                block.name
                                if hasattr(block, "name")
                                else block.get("name")
                            )
                            tool_input = (
                                block.input
                                if hasattr(block, "input")
                                else block.get("input", {})
                            )
                            tool_calls.append(
                                {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_input)
                                        if isinstance(tool_input, dict)
                                        else str(tool_input),
                                    },
                                }
                            )

                    msg_dict: Dict[str, Any] = {"role": "assistant"}
                    if text_parts:
                        msg_dict["content"] = "\n".join(text_parts)
                    else:
                        msg_dict["content"] = ""
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls
                    if reasoning_parts:
                        msg_dict["reasoning_content"] = "\n".join(reasoning_parts)
                    result.append(msg_dict)

                elif role == "user":
                    text_parts = []
                    tool_results = []

                    for block in content:
                        block_type = (
                            block.type
                            if hasattr(block, "type")
                            else block.get("type")
                            if isinstance(block, dict)
                            else None
                        )

                        if block_type == "text":
                            text = (
                                block.text
                                if hasattr(block, "text")
                                else block.get("text", "")
                            )
                            text_parts.append(text)
                        elif block_type == "tool_result":
                            tool_use_id = (
                                block.tool_use_id
                                if hasattr(block, "tool_use_id")
                                else block.get("tool_use_id")
                            )
                            tool_content = (
                                block.content
                                if hasattr(block, "content")
                                else block.get("content", "")
                            )
                            if isinstance(tool_content, list):
                                tool_content = "\n".join(
                                    item.get("text", str(item))
                                    if isinstance(item, dict)
                                    else str(item)
                                    for item in tool_content
                                )
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": str(tool_content)
                                    if tool_content
                                    else "",
                                }
                            )

                    for tr in tool_results:
                        result.append(tr)

                    if text_parts:
                        result.append(
                            {"role": "user", "content": "\n".join(text_parts)}
                        )
            else:
                result.append({"role": role, "content": str(content)})

        return result

    def _convert_tools(self, tools) -> list:
        """Convert Anthropic tools to OpenAI format."""
        result = []
        for tool in tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )
        return result

    def _build_request_body(self, request_data: Any, stream: bool = False) -> dict:
        """Build the request body for NVIDIA NIM API."""
        messages = self._convert_messages(request_data.messages)

        # Handle system message
        if request_data.system:
            if isinstance(request_data.system, str):
                messages.insert(0, {"role": "system", "content": request_data.system})
            elif isinstance(request_data.system, list):
                system_text = ""
                for block in request_data.system:
                    if hasattr(block, "type") and block.type == "text":
                        system_text += block.text + "\n\n"
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_text += block.get("text", "") + "\n\n"
                if system_text:
                    messages.insert(
                        0, {"role": "system", "content": system_text.strip()}
                    )

        body = {
            "model": request_data.model,
            "messages": messages,
            "max_tokens": request_data.max_tokens,
            "stream": stream,
        }

        if request_data.temperature is not None:
            body["temperature"] = request_data.temperature
        if request_data.top_p is not None:
            body["top_p"] = request_data.top_p
        if request_data.stop_sequences:
            body["stop"] = request_data.stop_sequences

        if request_data.tools:
            body["tools"] = self._convert_tools(request_data.tools)

        # Handle thinking/reasoning
        extra_body = request_data.extra_body.copy() if request_data.extra_body else {}
        if request_data.thinking and getattr(request_data.thinking, "enabled", True):
            # For some NIM models, we might need a specific flag or thinking type
            extra_body.setdefault("thinking", {"type": "enabled"})
            # Also ensure max_tokens is high enough or handle budget if needed in the future

        if extra_body:
            body["extra_body"] = extra_body

        # Merge env params
        for key, val in self._nim_params.items():
            if key not in body:
                body[key] = val

        return body

    async def stream_response_direct(
        self,
        request_data: Any,
        count_towards_limit: bool = True,
        input_tokens: int = 0,
    ) -> AsyncIterator[str]:
        """Stream response directly from NVIDIA NIM using httpx with thinking support."""
        if count_towards_limit:
            await self._enforce_rate_limit_async()

        body = self._build_request_body(request_data, stream=True)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # Send message_start
        message_id = f"msg_{uuid.uuid4()}"
        message_start = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request_data.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Track state
        accumulated_text = ""
        accumulated_reasoning = ""

        thinking_block_started = False
        thinking_index = -1

        text_block_started = False
        current_text_index = -1

        tool_calls_started = False
        next_block_index = 0

        tool_index_to_block = {}
        tool_index_to_content = {}
        finish_reason = None
        usage_info = None
        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=body,
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(
                    f"NVIDIA NIM API error: {response.status_code} - {error_text}"
                )
                raise Exception(f"NVIDIA NIM API error: {response.status_code}")

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk

                while "\n\n" in buffer:
                    event_end = buffer.index("\n\n")
                    event_data = buffer[:event_end]
                    buffer = buffer[event_end + 2 :]

                    if not event_data.strip():
                        continue

                    data_line = None
                    for line in event_data.split("\n"):
                        if line.startswith("data: "):
                            data_line = line[6:]
                            break

                    if not data_line or data_line == "[DONE]":
                        continue

                    try:
                        chunk_json = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue

                    if "choices" not in chunk_json or not chunk_json["choices"]:
                        if "usage" in chunk_json:
                            usage_info = chunk_json["usage"]
                        continue

                    choice = chunk_json["choices"][0]
                    delta = choice.get("delta", {})

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    if "usage" in chunk_json:
                        usage_info = chunk_json["usage"]

                    # 1. Handle Reasoning/Thinking
                    delta_reasoning = delta.get("reasoning_content")
                    if delta_reasoning:
                        # Close text if it was open
                        if text_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                            text_block_started = False

                        if not thinking_block_started:
                            # Start thinking block at next available index
                            thinking_index = next_block_index
                            next_block_index += 1
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': thinking_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                            thinking_block_started = True

                        accumulated_reasoning += delta_reasoning
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': thinking_index, 'delta': {'type': 'thinking_delta', 'thinking': delta_reasoning}})}\n\n"

                    # 2. Handle Text Content
                    delta_content = delta.get("content")
                    if delta_content:
                        # Log for debugging
                        logger.debug(f"[DEBUG] NVIDIA DELTA CONTENT: {delta_content[:100] if isinstance(delta_content, str) else type(delta_content)}")
                        # Close thinking if it was open
                        if thinking_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                            thinking_block_started = False

                        if not text_block_started:
                            current_text_index = next_block_index
                            next_block_index += 1
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': current_text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            text_block_started = True

                        accumulated_text += delta_content
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_text_index, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

                    # 3. Handle Tool Calls
                    if delta.get("tool_calls"):
                        # Close thinking or text if they were open
                        if thinking_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                            thinking_block_started = False
                        if text_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                            text_block_started = False

                        tool_calls_started = True

                        for tc in delta["tool_calls"]:
                            tc_index = tc.get("index", 0)
                            if tc_index < 0:
                                tc_index = len(tool_index_to_block)

                            if tc_index not in tool_index_to_block:
                                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                                tool_index_to_block[tc_index] = next_block_index
                                tool_index_to_content[tc_index] = ""
                                name = tc.get("function", {}).get("name", "")
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': next_block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                next_block_index += 1

                            args = tc.get("function", {}).get("arguments", "")
                            if args:
                                block_idx = tool_index_to_block[tc_index]
                                tool_index_to_content[tc_index] += args
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n"

        # Finalize and Close all blocks
        if thinking_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
        if tool_calls_started:
            for tc_index, block_idx in tool_index_to_block.items():
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"

        # Map finish reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
        }
        stop_reason = (
            stop_reason_map.get(finish_reason, "end_turn")
            if finish_reason
            else "end_turn"
        )

        # Usage
        if usage_info:
            final_output_tokens = usage_info.get("completion_tokens", 0)
        else:
            final_output_tokens = (
                len(accumulated_text) // 4
                + len(accumulated_reasoning) // 4
                + len(tool_index_to_content) * 50
            )

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': final_output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "[DONE]\n\n"

    async def call_api_direct(
        self, request_data: Any, count_towards_limit: bool = True
    ) -> Any:
        """Make non-streaming API call using httpx."""
        if count_towards_limit:
            await self._enforce_rate_limit_async()

        body = self._build_request_body(request_data, stream=False)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        return response.json()

    def convert_response_direct(
        self, response_json: dict, original_request: Any
    ) -> Any:
        """Convert raw JSON response to Anthropic format."""
        from ..models.request import MessagesResponse, Usage

        choice = response_json["choices"][0]
        message = choice["message"]
        content = []

        # Handle reasoning/thinking
        reasoning = message.get("reasoning_content")
        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning})

        if message.get("content"):
            raw_content = message["content"]
            if isinstance(raw_content, str):
                content.append({"type": "text", "text": raw_content})
            elif isinstance(raw_content, list):
                # Handle structured content: [{"type": "text", "text": "..."}]
                for item in raw_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content.append(item)

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = tc["function"].get("arguments", {})
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": args,
                    }
                )

        if not content:
            content.append({"type": "text", "text": ""})

        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
        }
        stop_reason = stop_reason_map.get(choice.get("finish_reason"), "end_turn")

        usage = response_json.get("usage", {})

        return MessagesResponse(
            id=response_json.get("id", f"msg_{uuid.uuid4()}"),
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
        )

    # Required abstract methods
    def prepare_request(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Convert Anthropic format to NIM/OpenAI format (sync; blocks on rate limit)."""
        if count_towards_limit:
            # Direct provider doesn't have a sync rate limit enforcement implemented,
            # but we can use the async one in a loop or just skip for now as this is mainly used for tokens.
            # For simplicity in this provider, we'll just skip sync rate limit here.
            pass
        return self._build_request_body(anthropic_request, stream=False)

    async def prepare_request_async(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Async version for preparing request."""
        if count_towards_limit:
            await self._enforce_rate_limit_async()
        return self._build_request_body(anthropic_request, stream=False)

    def prepare_streaming_request(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Same as prepare_request but with stream=True."""
        req = self.prepare_request(
            anthropic_request, count_towards_limit=count_towards_limit
        )
        req["stream"] = True
        return req

    async def prepare_streaming_request_async(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Async version for streaming."""
        req = await self.prepare_request_async(
            anthropic_request, count_towards_limit=count_towards_limit
        )
        req["stream"] = True
        return req

    def convert_response(
        self, litellm_response: Union[Dict, Any], original_request: Any
    ) -> Any:
        return self.convert_response_direct(litellm_response, original_request)

    async def stream_response(
        self, litellm_generator: AsyncIterator
    ) -> AsyncIterator[str]:
        # The direct provider uses its own streaming logic called by server.py
        # but we provide this to satisfy the abstract class requirements
        async for chunk in litellm_generator:
            yield str(chunk)

    def should_flatten_messages(self) -> bool:
        return True

    def supports_native_tool_results(self) -> bool:
        return True
