"""Anthropic Claude provider implementation."""

from typing import Dict, Any, Optional, AsyncIterator, Union
import logging
import json

from .base import BaseProvider, ProviderConfig

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider implementation with native format support."""

    def get_model_prefix(self) -> str:
        return "anthropic/"

    def prepare_request(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Native Anthropic format - minimal conversion."""
        # Import MessagesRequest type here to avoid circular import
        from ..models.request import MessagesRequest

        # Create LiteLLM request dict - keep Anthropic native format
        litellm_request = {
            "model": anthropic_request.model,
            "messages": self._convert_messages(anthropic_request.messages),
            "max_tokens": anthropic_request.max_tokens,
            "temperature": anthropic_request.temperature,
            "stream": False,
        }

        # Handle system message
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                litellm_request["system"] = anthropic_request.system
            elif isinstance(anthropic_request.system, list):
                # List of content blocks - LiteLLM may accept this, or we may need to flatten
                litellm_request["system"] = anthropic_request.system

        # Add optional parameters
        if anthropic_request.stop_sequences:
            litellm_request["stop"] = anthropic_request.stop_sequences
        if anthropic_request.top_p is not None:
            litellm_request["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            litellm_request["top_k"] = anthropic_request.top_k

        # Add thinking for Anthropic models
        if anthropic_request.thinking:
            litellm_request["thinking"] = anthropic_request.thinking

        # Add tools if present - convert to Anthropic format
        if anthropic_request.tools:
            litellm_request["tools"] = self._convert_tools(anthropic_request.tools)
            if anthropic_request.tool_choice:
                litellm_request["tool_choice"] = self._convert_tool_choice(
                    anthropic_request.tool_choice
                )

        return litellm_request

    def prepare_streaming_request(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Same as prepare_request but with stream=True."""
        req = self.prepare_request(anthropic_request)
        req["stream"] = True
        return req

    def convert_response(
        self, litellm_response: Union[Dict, Any], original_request: Any
    ) -> Any:
        """Convert Anthropic-style response (may already be in Anthropic format)."""
        from ..models.request import MessagesResponse, Usage
        import uuid
        import json

        # Extract response data
        response_dict = self._extract_response_dict(litellm_response)

        # Anthropic responses may come directly from LiteLLM in OpenAI format
        choices = response_dict.get("choices", [{}])
        message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
        content_list = message.get("content", [])
        content_text = message.get("content", "")
        tool_calls = message.get("tool_calls", None)
        finish_reason = (
            choices[0].get("finish_reason", "stop")
            if choices and len(choices) > 0
            else "stop"
        )
        usage_info = response_dict.get("usage", {})
        response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        content = []

        # Check if content is already a list of content blocks
        if isinstance(content_list, list) and content_list:
            # Already in Anthropic format, use as-is
            for block in content_list:
                if isinstance(block, dict):
                    content.append(block)
                else:
                    content.append({"type": "text", "text": str(block)})
        elif content_text and content_text != "":
            # String content - wrap in text block
            content.append({"type": "text", "text": content_text})

        # Handle tool calls (for Anthropic models, convert to tool_use blocks)
        if tool_calls:
            clean_model = original_request.model
            if clean_model.startswith("anthropic/"):
                clean_model = clean_model[len("anthropic/") :]

            is_claude_model = clean_model.startswith("claude-")

            if is_claude_model and tool_calls:
                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]

                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function = tool_call.get("function", {})
                        tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                        name = function.get("name", "")
                        arguments = function.get("arguments", {})
                    else:
                        function = getattr(tool_call, "function", None)
                        tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                        name = getattr(function, "name", "") if function else ""
                        arguments = (
                            getattr(function, "arguments", {}) if function else {}
                        )

                    # Convert string arguments to dict if needed
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}

                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": name,
                            "input": arguments,
                        }
                    )

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

        # Ensure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})

        return MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

    async def stream_response(
        self, litellm_generator: AsyncIterator, model: str = "claude-sonnet-4-20250514"
    ) -> AsyncIterator[str]:
        """Stream in Anthropic SSE format."""
        import json
        import uuid

        # Send message_start event with all required fields
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
                "usage": {"input_tokens": 0, "output_tokens": 1},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Send text block start
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        # Track state
        message_delta = {"stop_reason": None, "usage": None}
        accumulated_text = ""
        current_text_index = 0
        text_block_closed = False
        tool_calls_started = False
        next_block_index = 1
        # Track multiple tool calls by their index
        tool_index_to_block = {}  # tool index -> our block index
        tool_index_to_id = {}  # tool index -> tool_id
        tool_index_to_content = {}  # tool index -> accumulated arguments

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

            # Extract delta content
            delta_content = None
            delta_tool_calls = None

            if delta:
                if hasattr(delta, "content"):
                    delta_content = delta.content
                elif isinstance(delta, dict):
                    delta_content = delta.get("content")

                if hasattr(delta, "tool_calls"):
                    delta_tool_calls = delta.tool_calls
                elif isinstance(delta, dict):
                    delta_tool_calls = delta.get("tool_calls")

            # Handle tool calls
            if delta_tool_calls:
                if not tool_calls_started:
                    # Close text block
                    if not text_block_closed:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        text_block_closed = True
                    tool_calls_started = True

                if not isinstance(delta_tool_calls, list):
                    delta_tool_calls = [delta_tool_calls]

                for tool_call in delta_tool_calls:
                    # Extract tool call data - use 'index' to identify which tool
                    if isinstance(tool_call, dict):
                        tc_index = tool_call.get("index", 0)
                        function = tool_call.get("function", {})
                        tool_id = tool_call.get("id")
                        name = function.get("name") if function else None
                        arguments = function.get("arguments") if function else None
                    else:
                        tc_index = getattr(tool_call, "index", 0)
                        function = getattr(tool_call, "function", None)
                        tool_id = getattr(tool_call, "id", None)
                        name = getattr(function, "name", None) if function else None
                        arguments = (
                            getattr(function, "arguments", None) if function else None
                        )

                    # New tool - emit content_block_start
                    if tc_index not in tool_index_to_block:
                        # Generate ID if not provided
                        if not tool_id:
                            tool_id = f"tool_{uuid.uuid4()}"
                        tool_index_to_id[tc_index] = tool_id
                        tool_index_to_block[tc_index] = next_block_index
                        tool_index_to_content[tc_index] = ""
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': next_block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name or '', 'input': {}}})}\n\n"
                        next_block_index += 1

                    # Emit tool arguments
                    if arguments:
                        block_idx = tool_index_to_block[tc_index]
                        tool_index_to_content[tc_index] += (
                            arguments if isinstance(arguments, str) else str(arguments)
                        )
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': str(arguments)}})}\n\n"

            # Process text content
            if delta_content and delta_content != "" and not tool_calls_started:
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
                stop_reason = stop_reason_map.get(finish_reason, "end_turn")

                # Close any open tool blocks
                for tc_index, block_idx in tool_index_to_block.items():
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"

                # Close text block if still open
                if not text_block_closed and not tool_calls_started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
                    text_block_closed = True

                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': len(accumulated_text) // 4 + len(tool_index_to_content) * 50}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                yield "[DONE]\n\n"
                break

        # If we didn't get a finish_reason, close the response
        if not message_delta["stop_reason"]:
            # Close any open tool blocks
            for tc_index, block_idx in tool_index_to_block.items():
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
            # Close text block if still open
            if not text_block_closed and not tool_calls_started:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(accumulated_text) // 4 + 1}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "[DONE]\n\n"

    def should_flatten_messages(self) -> bool:
        return False

    # Helper methods
    def _convert_messages(self, messages):
        """Keep Anthropic format messages as-is."""
        converted = []
        for msg in messages:
            if isinstance(msg.content, str):
                converted.append({"role": msg.role, "content": msg.content})
            else:
                # Convert Pydantic models to dicts
                content_list = []
                for block in msg.content:
                    if isinstance(block, dict):
                        content_list.append(block)
                    else:
                        # Convert to dict
                        content_dict = {"type": block.type}
                        if block.type == "text":
                            content_dict["text"] = block.text
                        elif block.type == "image":
                            content_dict["source"] = block.source
                        elif block.type == "tool_use":
                            content_dict["id"] = block.id
                            content_dict["name"] = block.name
                            content_dict["input"] = block.input
                        elif block.type == "tool_result":
                            content_dict["tool_use_id"] = block.tool_use_id
                            content_dict["content"] = block.content
                        elif block.type == "thinking":
                            content_dict["thinking"] = block.thinking
                        content_list.append(content_dict)

                m = {"role": msg.role, "content": content_list}
                converted.append(m)
        return converted

    def _convert_tools(self, tools):
        """Keep Anthropic format tools as-is."""
        converted = []
        for tool in tools:
            if hasattr(tool, "dict"):
                tool_dict = tool.dict()
            else:
                tool_dict = tool

            converted.append(
                {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "input_schema": tool_dict.get("input_schema", {}),
                }
            )
        return converted

    def _convert_tool_choice(self, tool_choice):
        """Keep Anthropic format tool_choice as-is."""
        if hasattr(tool_choice, "dict"):
            return tool_choice.dict()
        return tool_choice

    def _extract_response_dict(self, response) -> Dict[str, Any]:
        """Extract dict from various response types."""
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump") and callable(
            getattr(response, "model_dump")
        ):
            return response.model_dump()
        if hasattr(response, "dict") and callable(getattr(response, "dict")):
            return response.dict()
        return response.__dict__ if hasattr(response, "__dict__") else {}
