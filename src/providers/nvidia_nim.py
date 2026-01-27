"""NVIDIA NIM provider with full pass-through and dynamic response parsing."""

import asyncio
import logging
from collections import deque
import threading
import time
import os
from typing import Dict, Any, Optional, AsyncIterator, Union
import json

from .base import BaseProvider, ProviderConfig

logger = logging.getLogger(__name__)


class NVIDIANIMProvider(BaseProvider):
    """NVIDIA NIM provider with full pass-through and dynamic response parsing."""

    # Rate limiting: shared across all NIM requests
    _rate_lock = threading.Condition()
    _request_timestamps = deque()

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._nim_params = self._load_nim_params()
        # Override model for NIM (add nvidia_nim/ prefix for LiteLLM)
        self._model_prefix = "nvidia_nim/"

    def _load_nim_params(self) -> Dict[str, Any]:
        """Load useful NIM parameters from environment variables."""
        params: Dict[str, Any] = {}

        # Core parameters
        if val := os.getenv("NVIDIA_NIM_TEMPERATURE"):
            params["temperature"] = float(val)
        if val := os.getenv("NVIDIA_NIM_TOP_P"):
            params["top_p"] = float(val)
        if val := os.getenv("NVIDIA_NIM_TOP_K"):
            params["top_k"] = int(val)
        if val := os.getenv("NVIDIA_NIM_MAX_TOKENS"):
            params["max_tokens"] = int(val)
        if val := os.getenv("NVIDIA_NIM_PRESENCE_PENALTY"):
            params["presence_penalty"] = float(val)
        if val := os.getenv("NVIDIA_NIM_FREQUENCY_PENALTY"):
            params["frequency_penalty"] = float(val)

        # Advanced parameters
        if val := os.getenv("NVIDIA_NIM_MIN_P"):
            params["min_p"] = float(val)
        if val := os.getenv("NVIDIA_NIM_REPETITION_PENALTY"):
            params["repetition_penalty"] = float(val)
        if val := os.getenv("NVIDIA_NIM_SEED"):
            params["seed"] = int(val)
        if val := os.getenv("NVIDIA_NIM_STOP"):
            if val.lower() != "none":
                params["stop"] = val.split(",") if "," in val else val

        # Flag parameters
        if val := os.getenv("NVIDIA_NIM_PARALLEL_TOOL_CALLS"):
            params["parallel_tool_calls"] = val.lower() == "true"
        if val := os.getenv("NVIDIA_NIM_RETURN_TOKENS_AS_TOKEN_IDS"):
            params["return_tokens_as_token_ids"] = val.lower() == "true"
        if val := os.getenv("NVIDIA_NIM_INCLUDE_STOP_STR_IN_OUTPUT"):
            params["include_stop_str_in_output"] = val.lower() == "true"
        if val := os.getenv("NVIDIA_NIM_IGNORE_EOS"):
            params["ignore_eos"] = val.lower() == "true"
        if val := os.getenv("NVIDIA_NIM_MIN_TOKENS"):
            params["min_tokens"] = int(val)

        # String parameters
        if val := os.getenv("NVIDIA_NIM_CHAT_TEMPLATE"):
            params["chat_template"] = val
        if val := os.getenv("NVIDIA_NIM_REQUEST_ID"):
            params["request_id"] = val

        # Thinking/reasoning (will be merged into extra_body)
        extra_body: Dict[str, Any] = {}
        if val := os.getenv("NVIDIA_NIM_REASONING_EFFORT"):
            extra_body["reasoning_effort"] = val
        if val := os.getenv("NVIDIA_NIM_INCLUDE_REASONING"):
            extra_body["include_reasoning"] = val.lower() == "true"

        if extra_body:
            params["extra_body"] = extra_body

        return params

    def get_model_prefix(self) -> str:
        return "nvidia_nim/"

    def _enforce_rate_limit(self):
        """Enforce configured rate limit (default 40 requests/minute).

        WARNING: This method uses blocking waits. In async contexts (like FastAPI),
        use _enforce_rate_limit_async() instead to avoid blocking the event loop.
        """
        # Check if we're in an async context and warn
        try:
            loop = asyncio.get_running_loop()
            if loop is not None:
                logger.warning(
                    "NVIDIA NIM: _enforce_rate_limit() called from async context! "
                    "This will block the event loop. Use prepare_request_async() instead."
                )
        except RuntimeError:
            pass  # No running loop, safe to use sync version

        rate_limit = self.config.rate_limit or 40
        now = time.time()
        with self._rate_lock:
            while (
                self._request_timestamps
                and now - self._request_timestamps[0] > self.config.rate_window
            ):
                self._request_timestamps.popleft()

            while len(self._request_timestamps) >= rate_limit:
                wait_time = self._request_timestamps[0] + self.config.rate_window - now
                if wait_time <= 0:
                    break
                # Use a short timeout to prevent indefinite blocking
                # This allows the method to re-check conditions periodically
                self._rate_lock.wait(min(wait_time, 0.5))
                now = time.time()
                while (
                    self._request_timestamps
                    and now - self._request_timestamps[0] > self.config.rate_window
                ):
                    self._request_timestamps.popleft()

            self._request_timestamps.append(now)
            logger.debug(
                f"NVIDIA NIM request recorded. Count: {len(self._request_timestamps)}/{rate_limit}"
            )

    async def _enforce_rate_limit_async(self):
        """Enforce rate limit using asyncio.sleep so the event loop is not blocked.

        This is the preferred method for async contexts (FastAPI, etc.).
        Uses non-blocking asyncio.sleep() instead of threading.Condition.wait().
        """
        rate_limit = self.config.rate_limit or 40
        max_retries = 120  # Prevent infinite loops (120 * 0.5s = 60s max wait)
        retry_count = 0

        while retry_count < max_retries:
            now = time.time()
            with self._rate_lock:
                # Clean up old timestamps outside the rate window
                while (
                    self._request_timestamps
                    and now - self._request_timestamps[0] > self.config.rate_window
                ):
                    self._request_timestamps.popleft()

                # Check if we can proceed
                if len(self._request_timestamps) < rate_limit:
                    self._request_timestamps.append(now)
                    logger.debug(
                        f"NVIDIA NIM request recorded. Count: {len(self._request_timestamps)}/{rate_limit}"
                    )
                    return

                # Calculate wait time
                wait_time = self._request_timestamps[0] + self.config.rate_window - now

            if wait_time <= 0:
                # Timestamps should have been cleaned, retry immediately
                retry_count += 1
                continue

            # Wait using non-blocking asyncio.sleep (capped at 1 second to stay responsive)
            actual_wait = min(wait_time, 1.0)
            logger.debug(
                f"NVIDIA NIM rate limit reached. Waiting {actual_wait:.2f}s (need {wait_time:.2f}s total)"
            )
            await asyncio.sleep(actual_wait)
            retry_count += 1

        # If we exceeded max retries, log warning but proceed anyway
        logger.warning(
            f"NVIDIA NIM rate limit wait exceeded {max_retries} retries, proceeding anyway"
        )
        with self._rate_lock:
            self._request_timestamps.append(time.time())

    def _prepare_request_impl(self, anthropic_request: Any) -> Dict[str, Any]:
        """Core request conversion without rate limiting. Used by prepare_request and prepare_request_async."""
        # Core conversion (messages)
        messages = self._convert_messages(anthropic_request.messages)

        # Handle system message
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                messages.insert(
                    0, {"role": "system", "content": anthropic_request.system}
                )
            elif isinstance(anthropic_request.system, list):
                system_text = ""
                for block in anthropic_request.system:
                    if hasattr(block, "type") and block.type == "text":
                        system_text += block.text + "\n\n"
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_text += block.get("text", "") + "\n\n"
                if system_text:
                    messages.insert(
                        0, {"role": "system", "content": system_text.strip()}
                    )

        # Start with standard request
        litellm_request = {
            "model": self._model_prefix
            + anthropic_request.model,  # Add nvidia_nim/ prefix
            "messages": messages,
            "max_tokens": anthropic_request.max_tokens,
            "stream": False,
        }

        # Add optional params from original request (highest priority)
        if anthropic_request.temperature is not None:
            litellm_request["temperature"] = anthropic_request.temperature
        if anthropic_request.top_p is not None:
            litellm_request["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            litellm_request["top_k"] = anthropic_request.top_k
        if anthropic_request.stop_sequences:
            litellm_request["stop"] = anthropic_request.stop_sequences

        # Add tools if present
        if anthropic_request.tools:
            litellm_request["tools"] = self._convert_tools(anthropic_request.tools)
            if anthropic_request.tool_choice:
                litellm_request["tool_choice"] = self._convert_tool_choice(
                    anthropic_request.tool_choice
                )

        # MERGE: env-provided NIM params (lower priority than request params)
        for key, val in self._nim_params.items():
            if key != "extra_body" and key not in litellm_request:
                litellm_request[key] = val

        # MERGE: extra_body - user extra_body is highest priority, then env extra_body
        user_extra_body = (
            anthropic_request.extra_body.copy() if anthropic_request.extra_body else {}
        )
        # Map anthropic_request.thinking to extra_body.thinking so GLM thinking is enabled
        if anthropic_request.thinking and getattr(
            anthropic_request.thinking, "enabled", True
        ):
            user_extra_body.setdefault("thinking", {"type": "enabled"})
        env_extra_body = self._nim_params.get("extra_body", {})
        if env_extra_body:
            for key, val in env_extra_body.items():
                if key not in user_extra_body:  # User-provided wins
                    user_extra_body[key] = val

        if user_extra_body:
            litellm_request["extra_body"] = user_extra_body

        return litellm_request

    def prepare_request(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Convert Anthropic format to NIM/OpenAI format with pass-through params (sync; blocks on rate limit)."""
        if count_towards_limit:
            self._enforce_rate_limit()
        return self._prepare_request_impl(anthropic_request)

    async def prepare_request_async(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Async version that uses asyncio.sleep for rate limiting instead of blocking."""
        if count_towards_limit:
            await self._enforce_rate_limit_async()
        return self._prepare_request_impl(anthropic_request)

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
        """Async version for streaming; uses non-blocking rate limit."""
        req = await self.prepare_request_async(
            anthropic_request, count_towards_limit=count_towards_limit
        )
        req["stream"] = True
        return req

    def convert_response(
        self, litellm_response: Union[Dict, Any], original_request: Any
    ) -> Any:
        """Convert NIM response to Anthropic format with dynamic parsing."""
        from ..models.request import MessagesResponse, Usage
        import uuid
        import json

        # Extract response data
        response_dict = self._extract_response_dict(litellm_response)

        choices = response_dict.get("choices", [{}])
        message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
        reasoning_content = message.get("reasoning_content")
        content_text = message.get("content", "")
        tool_calls = message.get("tool_calls")
        finish_reason = (
            choices[0].get("finish_reason", "stop")
            if choices and len(choices) > 0
            else "stop"
        )
        usage_info = response_dict.get("usage", {})
        response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        # Build Anthropic content blocks
        content = []

        # Prepend thinking block if reasoning_content is present (dynamic - only add if it exists)
        if reasoning_content:
            content.insert(0, {"type": "thinking", "thinking": reasoning_content})

        # Add text content
        if content_text:
            content.append({"type": "text", "text": content_text})

        # Handle tool calls (convert to Anthropic tool_use format)
        if tool_calls:
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for tc in tool_calls:
                if isinstance(tc, dict):
                    tool_id = tc.get(
                        "id", f"tool_{tc.get('function', {}).get('name', '')}"
                    )
                    tool_name = tc.get("function", {}).get("name", "")
                    tool_args = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args_dict = (
                            json.loads(tool_args)
                            if isinstance(tool_args, str)
                            else tool_args
                        )
                    except:
                        args_dict = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": args_dict,
                        }
                    )

        # Ensure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})

        # Extract usage
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        # Map finish_reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
        }
        stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        return MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
            reasoning_content=reasoning_content,  # Optional, only present if model supports it
        )

    async def stream_response(
        self,
        litellm_generator: AsyncIterator,
        input_tokens: int = 0,
        model: str = "claude-sonnet-4-20250514",
    ) -> AsyncIterator[str]:
        """Stream NIM response with dynamic reasoning_content parsing."""
        import json
        import uuid

        # Send message_start event
        message_id = f"msg_{uuid.uuid4()}"
        message_start = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Send initial content_block_start for thinking (index 0) so the client has a block during thinking phase
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"

        # Track state (0-based indices: initial thinking is 0, next block will be 1)
        message_delta = {"stop_reason": None, "usage": None}
        accumulated_text = ""
        accumulated_reasoning = ""
        thinking_index = 0
        thinking_block_started = True
        current_text_index = 0
        next_block_index = 1
        text_block_closed = True
        tool_calls_started = False
        # Track multiple tool calls by their tool_id (NIM returns index=-1 so we ignore it and use tool_id instead)
        tool_id_to_block = {}  # tool_id -> our block index
        tool_id_to_content = {}  # tool_id -> accumulated arguments

        # Track usage from LiteLLM
        actual_usage = None

        async for chunk in litellm_generator:
            # Extract choice and delta
            choice = None
            delta = None
            finish_reason = None

            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                finish_reason = getattr(choice, "finish_reason", None)
            elif isinstance(chunk, dict):
                choice = chunk.get("choices", [{}])[0] if chunk.get("choices") else {}
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

            # Extract usage if present
            if hasattr(chunk, "usage") and chunk.usage:
                actual_usage = chunk.usage
            elif isinstance(chunk, dict) and chunk.get("usage"):
                actual_usage = chunk.get("usage")

            # Extract delta content, delta reasoning, and delta tool_calls (DYNAMIC - parse whatever is present)
            delta_content = None
            delta_reasoning = None
            delta_tool_calls = None

            if delta:
                raw_content = (
                    delta.content
                    if hasattr(delta, "content")
                    else (delta.get("content") if isinstance(delta, dict) else None)
                )

                # Structured content: list of {type, text} or {type, content} parts
                if isinstance(raw_content, list):
                    for part in raw_content:
                        if isinstance(part, dict):
                            ptype = part.get("type") or part.get("role")
                            ptext = part.get("text") or part.get("content") or ""
                        else:
                            ptype = getattr(part, "type", None) or getattr(
                                part, "role", None
                            )
                            ptext = (
                                getattr(part, "text", None)
                                or getattr(part, "content", None)
                                or ""
                            )
                        if not ptext:
                            continue
                        if ptype in ("reasoning", "thinking"):
                            if not thinking_block_started:
                                if not text_block_closed:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                                    text_block_closed = True
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': next_block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                                thinking_index = next_block_index
                                next_block_index += 1
                                thinking_block_started = True
                            accumulated_reasoning += ptext
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': thinking_index, 'delta': {'type': 'thinking_delta', 'thinking': ptext}})}\n\n"
                        elif ptype in ("text", "output"):
                            if thinking_block_started:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                                thinking_block_started = False
                            if text_block_closed:
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': next_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                current_text_index = next_block_index
                                next_block_index += 1
                                text_block_closed = False
                            accumulated_text += ptext
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_text_index, 'delta': {'type': 'text_delta', 'text': ptext}})}\n\n"
                    continue

                # Scalar: content, reasoning, and tool_calls with fallback keys
                delta_content = raw_content if isinstance(raw_content, str) else None
                delta_reasoning = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    if not isinstance(delta, dict)
                    else (delta.get("reasoning_content") or delta.get("reasoning"))
                )
                delta_tool_calls = (
                    getattr(delta, "tool_calls", None)
                    if not isinstance(delta, dict)
                    else delta.get("tool_calls")
                )

            # Handle tool calls (before reasoning/text; takes precedence)
            if delta_tool_calls:
                if not tool_calls_started:
                    if thinking_block_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                        thinking_block_started = False
                    if not text_block_closed:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                        text_block_closed = True
                    tool_calls_started = True

                if not isinstance(delta_tool_calls, list):
                    delta_tool_calls = [delta_tool_calls]

                for tool_call in delta_tool_calls:
                    # Extract tool call data - use tool_id to identify which tool (NIM returns index=-1 so we ignore it)
                    if isinstance(tool_call, dict):
                        function = tool_call.get("function", {})
                        tool_id = tool_call.get("id")
                        name = function.get("name") if function else None
                        arguments = function.get("arguments") if function else None
                    else:
                        function = getattr(tool_call, "function", None)
                        tool_id = getattr(tool_call, "id", None)
                        name = getattr(function, "name", None) if function else None
                        arguments = (
                            getattr(function, "arguments", None) if function else None
                        )

                    # Generate ID if not provided (NIM sometimes omits tool_id in early chunks)
                    if not tool_id:
                        tool_id = f"tool_{uuid.uuid4()}"

                    # New tool - emit content_block_start
                    if tool_id not in tool_id_to_block:
                        tool_id_to_block[tool_id] = next_block_index
                        tool_id_to_content[tool_id] = ""
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': next_block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name or '', 'input': {}}})}\n\n"
                        next_block_index += 1

                    # Emit tool arguments
                    if arguments:
                        block_idx = tool_id_to_block[tool_id]
                        tool_id_to_content[tool_id] += (
                            arguments if isinstance(arguments, str) else str(arguments)
                        )
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': str(arguments)}})}\n\n"

            # Process reasoning_content if present (only if model returns it; skip when tool_calls started)
            if delta_reasoning and not tool_calls_started:
                if not thinking_block_started:
                    if not text_block_closed:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                        text_block_closed = True
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': next_block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                    thinking_index = next_block_index
                    next_block_index += 1
                    thinking_block_started = True
                accumulated_reasoning += delta_reasoning
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': thinking_index, 'delta': {'type': 'thinking_delta', 'thinking': delta_reasoning}})}\n\n"

                # If both reasoning and content in same chunk: close thinking, open text
                if delta_content and delta_content != "":
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                    thinking_block_started = False
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': next_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    current_text_index = next_block_index
                    next_block_index += 1
                    text_block_closed = False

            # Process text content (skip when tool_calls started)
            if delta_content and delta_content != "" and not tool_calls_started:
                # Close thinking if still open
                if thinking_block_started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                    thinking_block_started = False

                if text_block_closed:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': next_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    current_text_index = next_block_index
                    next_block_index += 1
                    text_block_closed = False

                accumulated_text += delta_content
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_text_index, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

            # Handle finish_reason
            if finish_reason and not message_delta["stop_reason"]:
                message_delta["stop_reason"] = finish_reason
                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "content_filter": "stop_sequence",
                }
                stop_reason = stop_reason_map.get(finish_reason, "end_turn")  # type: ignore[arg-type]

                # Close any open tool blocks
                for tool_id, block_idx in tool_id_to_block.items():
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                if thinking_block_started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                    thinking_block_started = False

                if not text_block_closed:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                    text_block_closed = True

                # Determine final usage
                final_usage = {
                    "output_tokens": (len(accumulated_text) // 4)
                    + (len(accumulated_reasoning) // 4)
                    + (len(tool_id_to_content) * 50)
                }
                if actual_usage:
                    if hasattr(actual_usage, "completion_tokens"):
                        final_usage["output_tokens"] = actual_usage.completion_tokens
                    elif isinstance(actual_usage, dict):
                        final_usage["output_tokens"] = actual_usage.get(
                            "completion_tokens", final_usage["output_tokens"]
                        )

                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': final_usage})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                yield "[DONE]\n"
                break

        # If we didn't get a finish_reason, close any open blocks
        if not message_delta["stop_reason"]:
            # Close any open tool blocks
            for tool_id, block_idx in tool_id_to_block.items():
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
            if thinking_block_started:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
            if not text_block_closed:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"

            # Determine final usage
            final_usage = {
                "output_tokens": (len(accumulated_text) // 4)
                + (len(accumulated_reasoning) // 4)
                + 1
            }
            if actual_usage:
                if hasattr(actual_usage, "completion_tokens"):
                    final_usage["output_tokens"] = actual_usage.completion_tokens
                elif isinstance(actual_usage, dict):
                    final_usage["output_tokens"] = actual_usage.get(
                        "completion_tokens", final_usage["output_tokens"]
                    )

            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': final_usage})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "[DONE]\n"

    def should_flatten_messages(self) -> bool:
        return True

    def supports_native_tool_results(self) -> bool:
        return True

    # Helper methods for message/tool conversion
    def _convert_messages(self, messages) -> list:
        """Convert Anthropic messages to OpenAI format with tool_result handling."""
        converted = []

        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                m = {"role": msg.role, "content": content}
                converted.append(m)
            else:
                # Helper to get block type
                def _block_type(b):
                    return getattr(b, "type", None) or (
                        b.get("type") if isinstance(b, dict) else None
                    )

                # user with only tool_result -> role "tool" messages
                is_only_tool_result = (
                    msg.role == "user"
                    and len(content) > 0
                    and all(_block_type(b) == "tool_result" for b in content)
                )
                if is_only_tool_result:
                    for block in content:
                        bid = getattr(block, "tool_use_id", None) or (
                            block.get("tool_use_id") if isinstance(block, dict) else ""
                        )
                        bcontent = (
                            getattr(block, "content", None)
                            if not isinstance(block, dict)
                            else block.get("content")
                        )
                        flat = self._parse_tool_result_content(bcontent)
                        converted.append(
                            {"role": "tool", "tool_call_id": bid, "content": flat}
                        )
                    continue

                # assistant with list content -> content string, tool_calls, reasoning_content
                if msg.role == "assistant" and isinstance(content, list):
                    text_parts = []
                    tool_calls = []
                    reasoning_parts = []
                    for block in content:
                        btype = _block_type(block)
                        if btype == "text":
                            t = getattr(block, "text", None) or (
                                block.get("text") if isinstance(block, dict) else ""
                            )
                            text_parts.append(t)
                        elif btype == "thinking":
                            th = getattr(block, "thinking", None) or (
                                block.get("thinking") if isinstance(block, dict) else ""
                            )
                            reasoning_parts.append(th)
                        elif btype == "tool_use":
                            bid = getattr(block, "id", None) or (
                                block.get("id") if isinstance(block, dict) else ""
                            )
                            bname = getattr(block, "name", None) or (
                                block.get("name") if isinstance(block, dict) else ""
                            )
                            binput = getattr(block, "input", None) or (
                                block.get("input", {})
                                if isinstance(block, dict)
                                else {}
                            )
                            args = (
                                json.dumps(binput)
                                if isinstance(binput, dict)
                                else str(binput)
                            )
                            tool_calls.append(
                                {
                                    "id": bid,
                                    "type": "function",
                                    "function": {"name": bname, "arguments": args},
                                }
                            )
                    m = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                    }
                    if tool_calls:
                        m["tool_calls"] = tool_calls
                    reasoning_content = getattr(msg, "reasoning_content", None) or (
                        "\n".join(reasoning_parts) if reasoning_parts else None
                    )
                    if reasoning_content:
                        m["reasoning_content"] = reasoning_content
                    if getattr(msg, "thinking_blocks", None):
                        m["thinking_blocks"] = msg.thinking_blocks
                    converted.append(m)
                    continue

                # Regular handling for other message types
                text_content = ""
                reasoning_parts = []
                for block in content:
                    btype = _block_type(block)
                    if btype == "text":
                        t = getattr(block, "text", None) or (
                            block.get("text", "") if isinstance(block, dict) else ""
                        )
                        text_content += t + "\n"
                    elif btype == "thinking":
                        th = getattr(block, "thinking", None) or (
                            block.get("thinking") if isinstance(block, dict) else ""
                        )
                        reasoning_parts.append(th)
                    elif btype == "tool_use":
                        binput = getattr(block, "input", None) or (
                            block.get("input", {}) if isinstance(block, dict) else {}
                        )
                        text_content += f"[Tool: {getattr(block, 'name', '')}]\n{json.dumps(binput)}\n\n"
                    elif btype == "tool_result":
                        tool_id = getattr(block, "tool_use_id", None) or (
                            block.get("tool_use_id", "")
                            if isinstance(block, dict)
                            else ""
                        )
                        bcontent = (
                            getattr(block, "content", None)
                            if not isinstance(block, dict)
                            else block.get("content")
                        )
                        text_content += f"Tool result for {tool_id}:\n{self._parse_tool_result_content(bcontent)}\n"

                m = {"role": msg.role, "content": text_content.strip() or "..."}
                reasoning_content = getattr(msg, "reasoning_content", None) or (
                    "\n".join(reasoning_parts) if reasoning_parts else None
                )
                if reasoning_content:
                    m["reasoning_content"] = reasoning_content
                if getattr(msg, "thinking_blocks", None):
                    m["thinking_blocks"] = msg.thinking_blocks
                converted.append(m)

        return converted

    def _parse_tool_result_content(self, content):
        """Helper function to properly parse and normalize tool result content."""
        if content is None:
            return "No content provided"
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            result = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    result += item.get("text", "") + "\n"
                elif isinstance(item, str):
                    result += item + "\n"
                elif isinstance(item, dict):
                    if "text" in item:
                        result += item.get("text", "") + "\n"
                    else:
                        try:
                            import json

                            result += json.dumps(item) + "\n"
                        except:
                            result += str(item) + "\n"
                else:
                    try:
                        result += str(item) + "\n"
                    except:
                        result += "Unparseable content\n"
            return result.strip()
        if isinstance(content, dict):
            if content.get("type") == "text":
                return content.get("text", "")
            try:
                import json

                return json.dumps(content)
            except:
                return str(content)
        try:
            return str(content)
        except:
            return "Unparseable content"

    def _convert_tools(self, tools):
        """Convert Anthropic tools to OpenAI function format."""
        openai_tools = []
        for tool in tools:
            if hasattr(tool, "dict"):
                tool_dict = tool.dict()
            else:
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                    logger.error(f"Could not convert tool to dict: {tool}")
                    continue

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": tool_dict.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    def _convert_tool_choice(self, tool_choice):
        """Convert Anthropic tool_choice to OpenAI format."""
        if hasattr(tool_choice, "dict"):
            tool_choice_dict = tool_choice.dict()
        else:
            tool_choice_dict = tool_choice

        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            return "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            return {"type": "function", "function": {"name": tool_choice_dict["name"]}}
        return "auto"

    def _extract_response_dict(self, response) -> Dict[str, Any]:
        """Extract dict from various response types (ModelResponse, dict, etc.)."""
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump") and callable(
            getattr(response, "model_dump")
        ):
            return response.model_dump()
        if hasattr(response, "dict") and callable(getattr(response, "dict")):
            return response.dict()
        return response.__dict__ if hasattr(response, "__dict__") else {}
