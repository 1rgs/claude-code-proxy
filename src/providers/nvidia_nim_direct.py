"""Direct NVIDIA NIM provider using raw httpx - bypasses all libraries."""
import asyncio
import logging
import os
import json
import uuid
import time
import threading
from collections import deque
from typing import Dict, Any, Optional, AsyncIterator

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
        self._base_url = (config.base_url or os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
        self._nim_params = self._load_nim_params()

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
        """Enforce rate limit using asyncio.sleep."""
        rate_limit = self.config.rate_limit or 40
        rate_window = self.config.rate_window or 60
        now = time.time()

        while True:
            with self._rate_lock:
                while self._request_timestamps and now - self._request_timestamps[0] > rate_window:
                    self._request_timestamps.popleft()
                if len(self._request_timestamps) < rate_limit:
                    self._request_timestamps.append(now)
                    return
                wait_time = self._request_timestamps[0] + rate_window - now
            if wait_time <= 0:
                now = time.time()
                continue
            await asyncio.sleep(wait_time)
            now = time.time()

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

                    for block in content:
                        block_type = block.type if hasattr(block, 'type') else block.get('type') if isinstance(block, dict) else None

                        if block_type == "text":
                            text = block.text if hasattr(block, 'text') else block.get('text', '')
                            text_parts.append(text)
                        elif block_type == "tool_use":
                            tool_id = block.id if hasattr(block, 'id') else block.get('id')
                            tool_name = block.name if hasattr(block, 'name') else block.get('name')
                            tool_input = block.input if hasattr(block, 'input') else block.get('input', {})
                            tool_calls.append({
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
                                }
                            })

                    msg_dict = {"role": "assistant"}
                    if text_parts:
                        msg_dict["content"] = "\n".join(text_parts)
                    else:
                        msg_dict["content"] = None
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls
                    result.append(msg_dict)

                elif role == "user":
                    text_parts = []
                    tool_results = []

                    for block in content:
                        block_type = block.type if hasattr(block, 'type') else block.get('type') if isinstance(block, dict) else None

                        if block_type == "text":
                            text = block.text if hasattr(block, 'text') else block.get('text', '')
                            text_parts.append(text)
                        elif block_type == "tool_result":
                            tool_use_id = block.tool_use_id if hasattr(block, 'tool_use_id') else block.get('tool_use_id')
                            tool_content = block.content if hasattr(block, 'content') else block.get('content', '')
                            if isinstance(tool_content, list):
                                tool_content = "\n".join(
                                    item.get('text', str(item)) if isinstance(item, dict) else str(item)
                                    for item in tool_content
                                )
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": str(tool_content) if tool_content else ""
                            })

                    for tr in tool_results:
                        result.append(tr)

                    if text_parts:
                        result.append({"role": "user", "content": "\n".join(text_parts)})
            else:
                result.append({"role": role, "content": str(content)})

        return result

    def _convert_tools(self, tools) -> list:
        """Convert Anthropic tools to OpenAI format."""
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                }
            })
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
                    if hasattr(block, 'type') and block.type == "text":
                        system_text += block.text + "\n\n"
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_text += block.get("text", "") + "\n\n"
                if system_text:
                    messages.insert(0, {"role": "system", "content": system_text.strip()})

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

        # Merge env params
        for key, val in self._nim_params.items():
            if key not in body:
                body[key] = val

        return body

    async def stream_response_direct(self, request_data: Any) -> AsyncIterator[str]:
        """Stream response directly from NVIDIA NIM using httpx."""
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
                "usage": {"input_tokens": 0, "output_tokens": 1}
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Send initial text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        # Track state
        accumulated_text = ""
        current_text_index = 0
        text_block_closed = False
        tool_calls_started = False
        next_block_index = 1
        tool_index_to_block = {}
        tool_index_to_id = {}
        tool_index_to_content = {}
        finish_reason = None

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=body,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"NVIDIA NIM API error: {response.status_code} - {error_text}")
                    raise Exception(f"NVIDIA NIM API error: {response.status_code}")

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk

                    # Process complete SSE events
                    while "\n\n" in buffer:
                        event_end = buffer.index("\n\n")
                        event_data = buffer[:event_end]
                        buffer = buffer[event_end + 2:]

                        # Parse SSE event
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
                            continue

                        choice = chunk_json["choices"][0]
                        delta = choice.get("delta", {})

                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                        # Handle tool calls
                        if delta.get("tool_calls"):
                            if not tool_calls_started:
                                if not text_block_closed:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                                    text_block_closed = True
                                tool_calls_started = True

                            for tc in delta["tool_calls"]:
                                tc_index = tc.get("index", 0)
                                if tc_index < 0:
                                    tc_index = len(tool_index_to_block)

                                if tc_index not in tool_index_to_block:
                                    tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                                    tool_index_to_id[tc_index] = tool_id
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

                        # Handle text content
                        if delta.get("content") and not tool_calls_started:
                            text = delta["content"]
                            accumulated_text += text
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_text_index, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

        # Close all blocks
        if tool_calls_started:
            for tc_index, block_idx in tool_index_to_block.items():
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"

        if not text_block_closed:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"

        # Map finish reason
        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
        stop_reason = stop_reason_map.get(finish_reason, "end_turn") if finish_reason else "end_turn"

        output_tokens = len(accumulated_text) // 4 + len(tool_index_to_content) * 50
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "[DONE]\n\n"

    async def call_api_direct(self, request_data: Any) -> Any:
        """Make non-streaming API call using httpx."""
        await self._enforce_rate_limit_async()

        body = self._build_request_body(request_data, stream=False)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            return response.json()

    def convert_response_direct(self, response_json: dict, original_request: Any) -> Any:
        """Convert raw JSON response to Anthropic format."""
        from ..models.request import MessagesResponse, Usage

        choice = response_json["choices"][0]
        message = choice["message"]
        content = []

        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except:
                    args = {}
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": args
                })

        if not content:
            content.append({"type": "text", "text": ""})

        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
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
                output_tokens=usage.get("completion_tokens", 0)
            )
        )

    # Required abstract methods (not used for direct provider)
    def prepare_request(self, anthropic_request: Any) -> Dict[str, Any]:
        return {}

    def prepare_streaming_request(self, anthropic_request: Any) -> Dict[str, Any]:
        return {}

    def convert_response(self, response: Any, original_request: Any) -> Any:
        return self.convert_response_direct(response, original_request)

    async def stream_response(self, litellm_generator: AsyncIterator) -> AsyncIterator[str]:
        raise NotImplementedError("Direct provider uses stream_response_direct")

    def should_flatten_messages(self) -> bool:
        return True

    def supports_native_tool_results(self) -> bool:
        return True
