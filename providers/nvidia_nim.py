"""NVIDIA NIM provider - converts Anthropic format to OpenAI format for NIM."""

import asyncio
import logging
import os
import json
import uuid
from typing import Dict, Any, AsyncIterator, Optional
import time
import threading
from collections import deque
import httpx
from .base import BaseProvider, ProviderConfig

logger = logging.getLogger(__name__)


class NvidiaNimProvider(BaseProvider):
    """NVIDIA NIM provider using direct httpx requests."""

    _rate_lock = threading.Condition()
    _request_timestamps = deque()

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._api_key = config.api_key or os.getenv("NVIDIA_NIM_API_KEY", "")
        self._base_url = (config.base_url or os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
        self._nim_params = self._load_nim_params()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0, read=60.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    def _load_nim_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if val := os.getenv("NVIDIA_NIM_TEMPERATURE"):
            params["temperature"] = float(val)
        if val := os.getenv("NVIDIA_NIM_TOP_P"):
            params["top_p"] = float(val)
        if val := os.getenv("NVIDIA_NIM_MAX_TOKENS"):
            params["max_tokens"] = int(val)
        return params

    async def _enforce_rate_limit(self):
        rate_limit = self.config.rate_limit or 40
        rate_window = self.config.rate_window or 60
        max_retries = 120

        for _ in range(max_retries):
            now = time.time()
            with self._rate_lock:
                while self._request_timestamps and now - self._request_timestamps[0] > rate_window:
                    self._request_timestamps.popleft()

                if len(self._request_timestamps) < rate_limit:
                    self._request_timestamps.append(now)
                    return

                wait_time = self._request_timestamps[0] + rate_window - now

            if wait_time <= 0:
                continue

            await asyncio.sleep(min(wait_time, 1.0))

        with self._rate_lock:
            self._request_timestamps.append(time.time())

    def _convert_messages(self, messages) -> list:
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
                        block_type = block.type if hasattr(block, "type") else block.get("type") if isinstance(block, dict) else None

                        if block_type == "text":
                            text = block.text if hasattr(block, "text") else block.get("text", "")
                            text_parts.append(text)
                        elif block_type == "thinking":
                            thinking = block.thinking if hasattr(block, "thinking") else block.get("thinking", "")
                            reasoning_parts.append(thinking)
                        elif block_type == "tool_use":
                            tool_id = block.id if hasattr(block, "id") else block.get("id")
                            tool_name = block.name if hasattr(block, "name") else block.get("name")
                            tool_input = block.input if hasattr(block, "input") else block.get("input", {})
                            tool_calls.append({
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                                },
                            })

                    msg_dict: Dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else ""}
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls
                    if reasoning_parts:
                        msg_dict["reasoning_content"] = "\n".join(reasoning_parts)
                    result.append(msg_dict)

                elif role == "user":
                    text_parts = []
                    tool_results = []

                    for block in content:
                        block_type = block.type if hasattr(block, "type") else block.get("type") if isinstance(block, dict) else None

                        if block_type == "text":
                            text = block.text if hasattr(block, "text") else block.get("text", "")
                            text_parts.append(text)
                        elif block_type == "tool_result":
                            tool_use_id = block.tool_use_id if hasattr(block, "tool_use_id") else block.get("tool_use_id")
                            tool_content = block.content if hasattr(block, "content") else block.get("content", "")
                            if isinstance(tool_content, list):
                                tool_content = "\n".join(
                                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                                    for item in tool_content
                                )
                            tool_results.append({"role": "tool", "tool_call_id": tool_use_id, "content": str(tool_content) if tool_content else ""})

                    for tr in tool_results:
                        result.append(tr)

                    if text_parts:
                        result.append({"role": "user", "content": "\n".join(text_parts)})
            else:
                result.append({"role": role, "content": str(content)})

        return result

    def _convert_tools(self, tools) -> list:
        return [
            {"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}}
            for tool in tools
        ]

    def _build_request_body(self, request_data: Any, stream: bool = False) -> dict:
        messages = self._convert_messages(request_data.messages)

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
                    messages.insert(0, {"role": "system", "content": system_text.strip()})

        body = {"model": request_data.model, "messages": messages, "max_tokens": request_data.max_tokens, "stream": stream}

        if request_data.temperature is not None:
            body["temperature"] = request_data.temperature
        if request_data.top_p is not None:
            body["top_p"] = request_data.top_p
        if request_data.stop_sequences:
            body["stop"] = request_data.stop_sequences
        if request_data.tools:
            body["tools"] = self._convert_tools(request_data.tools)

        extra_body = request_data.extra_body.copy() if request_data.extra_body else {}
        if request_data.thinking and getattr(request_data.thinking, "enabled", True):
            extra_body.setdefault("thinking", {"type": "enabled"})
            extra_body.setdefault("reasoning_split", True)

        if extra_body:
            body["extra_body"] = extra_body

        for key, val in self._nim_params.items():
            if key not in body:
                body[key] = val

        return body

    async def stream_response(self, request_data: Any, input_tokens: int = 0) -> AsyncIterator[str]:
        await self._enforce_rate_limit()

        body = self._build_request_body(request_data, stream=True)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", "Accept": "text/event-stream"}

        message_id = f"msg_{uuid.uuid4()}"
        message_start = {
            "type": "message_start",
            "message": {
                "id": message_id, "type": "message", "role": "assistant", "content": [],
                "model": request_data.model, "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

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
        stream_done = False
        error_occurred = False
        error_message = ""

        try:
            async with self._client.stream("POST", f"{self._base_url}/chat/completions", headers=headers, json=body) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_occurred = True
                    error_message = f"API error {response.status_code}: {error_text.decode('utf-8', errors='replace')}"
                else:
                    buffer = ""
                    async for chunk in response.aiter_text():
                        if stream_done:
                            break
                        buffer += chunk

                        while "\n\n" in buffer:
                            event_end = buffer.index("\n\n")
                            event_data = buffer[:event_end]
                            buffer = buffer[event_end + 2:]

                            if not event_data.strip():
                                continue

                            data_line = None
                            for line in event_data.split("\n"):
                                if line.startswith("data: "):
                                    data_line = line[6:]
                                    break

                            if not data_line or data_line == "[DONE]":
                                if data_line == "[DONE]":
                                    stream_done = True
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

                            # Handle reasoning/thinking
                            delta_reasoning = delta.get("reasoning_content")
                            if not delta_reasoning:
                                reasoning_details = delta.get("reasoning_details")
                                if reasoning_details and isinstance(reasoning_details, list):
                                    delta_reasoning = "".join(item.get("text", "") for item in reasoning_details if isinstance(item, dict))

                            if delta_reasoning:
                                if text_block_started:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                                    text_block_started = False

                                if not thinking_block_started:
                                    thinking_index = next_block_index
                                    next_block_index += 1
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': thinking_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                                    thinking_block_started = True

                                accumulated_reasoning += delta_reasoning
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': thinking_index, 'delta': {'type': 'thinking_delta', 'thinking': delta_reasoning}})}\n\n"

                            # Handle text content
                            delta_content = delta.get("content")
                            if delta_content:
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

                            # Handle tool calls
                            if delta.get("tool_calls"):
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

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_occurred = True
            error_message = str(e)

        if error_occurred:
            error_block_index = next_block_index
            next_block_index += 1
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': error_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': error_block_index, 'delta': {'type': 'text_delta', 'text': error_message}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': error_block_index})}\n\n"

        if thinking_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
        if tool_calls_started:
            for tc_index, block_idx in tool_index_to_block.items():
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"

        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "stop_sequence"}
        stop_reason = stop_reason_map.get(finish_reason, "end_turn") if finish_reason else "end_turn"

        final_output_tokens = usage_info.get("completion_tokens", 0) if usage_info else (len(accumulated_text) // 4 + len(accumulated_reasoning) // 4 + len(tool_index_to_content) * 50)

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': final_output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "[DONE]\n\n"

    async def complete(self, request_data: Any) -> dict:
        await self._enforce_rate_limit()

        body = self._build_request_body(request_data, stream=False)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        response = await self._client.post(f"{self._base_url}/chat/completions", headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    def convert_response(self, response_json: dict, original_request: Any) -> dict:
        choice = response_json["choices"][0]
        message = choice["message"]
        content = []

        reasoning = message.get("reasoning_content")
        if not reasoning:
            reasoning_details = message.get("reasoning_details")
            if reasoning_details and isinstance(reasoning_details, list):
                reasoning = "\n".join(item.get("text", "") for item in reasoning_details if isinstance(item, dict))
        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning})

        if message.get("content"):
            raw_content = message["content"]
            if isinstance(raw_content, str):
                content.append({"type": "text", "text": raw_content})
            elif isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content.append(item)

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = tc["function"].get("arguments", {})
                content.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": args})

        if not content:
            content.append({"type": "text", "text": ""})

        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "stop_sequence"}
        stop_reason = stop_reason_map.get(choice.get("finish_reason"), "end_turn")

        usage = response_json.get("usage", {})

        return {
            "id": response_json.get("id", f"msg_{uuid.uuid4()}"),
            "type": "message",
            "role": "assistant",
            "model": original_request.model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
