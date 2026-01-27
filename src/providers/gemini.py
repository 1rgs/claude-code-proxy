"""Google Gemini provider implementation."""

from typing import Dict, Any, Optional, AsyncIterator, Union
import logging
import json

from .base import BaseProvider, ProviderConfig

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Google Gemini provider implementation."""

    def get_model_prefix(self) -> str:
        return "gemini/"

    def prepare_request(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Convert with Gemini schema cleaning."""
        # Import MessagesRequest type here to avoid circular import
        from ..models.request import MessagesRequest

        # Set up messages (same as Openai - flatten to OpenAI format)
        messages = []

        # Add system message if present
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                messages.append({"role": "system", "content": anthropic_request.system})
            elif isinstance(anthropic_request.system, list):
                system_text = ""
                for block in anthropic_request.system:
                    if hasattr(block, "type") and block.type == "text":
                        system_text += block.text + "\n\n"
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_text += block.get("text", "") + "\n\n"
                if system_text:
                    messages.append({"role": "system", "content": system_text.strip()})

        # Add conversation messages with proper tool handling
        for msg in anthropic_request.messages:
            content = msg.content
            if isinstance(content, str):
                m = {"role": msg.role, "content": content}
                messages.append(m)
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
                        messages.append(
                            {"role": "tool", "tool_call_id": bid, "content": flat}
                        )
                    continue

                # assistant with list content -> content string + tool_calls
                if msg.role == "assistant" and isinstance(content, list):
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        btype = _block_type(block)
                        if btype == "text":
                            t = getattr(block, "text", None) or (
                                block.get("text") if isinstance(block, dict) else ""
                            )
                            text_parts.append(t)
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
                    messages.append(m)
                    continue

                # Regular handling for other message types - flatten to text
                text_content = ""
                for block in content:
                    btype = _block_type(block)
                    if btype == "text":
                        t = getattr(block, "text", None) or (
                            block.get("text", "") if isinstance(block, dict) else ""
                        )
                        text_content += t + "\n"
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
                    elif btype == "image":
                        text_content += "[Image]\n"

                m = {"role": msg.role, "content": text_content.strip() or "..."}
                messages.append(m)

        # Cap max_tokens for Gemini models
        max_tokens = min(anthropic_request.max_tokens, 16384)

        # Create LiteLLM request dict
        litellm_request = {
            "model": anthropic_request.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": anthropic_request.temperature,
            "stream": False,
        }

        # Add optional parameters
        if anthropic_request.stop_sequences:
            litellm_request["stop"] = anthropic_request.stop_sequences
        if anthropic_request.top_p is not None:
            litellm_request["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            litellm_request["top_k"] = anthropic_request.top_k

        # Add tools if present - with schema cleaning
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

    async def prepare_request_async(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Async version for non-blocking request preparation."""
        # Gemini provider has no blocking operations, so we can just return directly
        return self.prepare_request(anthropic_request, count_towards_limit)

    async def prepare_streaming_request_async(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Async version for non-blocking streaming request preparation."""
        req = await self.prepare_request_async(anthropic_request, count_towards_limit)
        req["stream"] = True
        return req

    def convert_response(
        self, litellm_response: Union[Dict, Any], original_request: Any
    ) -> Any:
        """Convert Gemini response to Anthropic format."""
        from ..models.request import MessagesResponse, Usage
        import uuid
        import json

        # Extract response data
        response_dict = self._extract_response_dict(litellm_response)

        choices = response_dict.get("choices", [{}])
        message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
        content_text = message.get("content", "")
        tool_calls = message.get("tool_calls", None)
        finish_reason = (
            choices[0].get("finish_reason", "stop")
            if choices and len(choices) > 0
            else "stop"
        )
        usage_info = response_dict.get("usage", {})
        response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        # Build Anthropic content blocks
        content = []

        # Add text content
        if content_text and content_text != "":
            content.append({"type": "text", "text": content_text})

        # Convert tool calls to proper Anthropic tool_use blocks
        if tool_calls:
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = (
                        getattr(function, "arguments", "{}") if function else "{}"
                    )

                # Parse arguments to dict
                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                    except json.JSONDecodeError:
                        args_dict = {"raw": arguments}
                else:
                    args_dict = arguments

                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": args_dict,
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
        self,
        litellm_generator: AsyncIterator,
        input_tokens: int = 0,
        model: str = "claude-sonnet-4-20250514",
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
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Send text block start
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        # Track state
        message_delta = {"stop_reason": None, "usage": None}
        accumulated_text = ""
        current_text_index = 0  # Track actual text block index
        text_block_closed = False
        tool_calls_started = False
        next_block_index = 1
        # Track multiple tool calls by their index
        tool_index_to_block = {}  # tool index -> our block index
        tool_index_to_id = {}  # tool index -> tool_id
        tool_index_to_content = {}  # tool index -> accumulated arguments
        # Track fallback index generation for negative tool indices (e.g., when NVIDIA NIM returns -1)
        next_tool_fallback_index = 0

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

            # Extract delta content and tool calls
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
                    # Close text block first
                    if not text_block_closed:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_text_index})}\n\n"
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

                    # Handle negative indices (e.g., when NVIDIA NIM returns -1)
                    if tc_index < 0:
                        tc_index = next_tool_fallback_index
                        next_tool_fallback_index += 1

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

            # Process text content (skip when tool_calls started)
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

                # Determine final usage
                final_usage = {
                    "output_tokens": len(accumulated_text) // 4
                    + len(tool_index_to_content) * 50
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

            # Determine final usage
            final_usage = {"output_tokens": len(accumulated_text) // 4 + 1}
            if actual_usage:
                if hasattr(actual_usage, "completion_tokens"):
                    final_usage["output_tokens"] = actual_usage.completion_tokens
                elif isinstance(actual_usage, dict):
                    final_usage["output_tokens"] = actual_usage.get(
                        "completion_tokens", final_usage["output_tokens"]
                    )

            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': final_usage})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "[DONE]\n\n"

    def should_flatten_messages(self) -> bool:
        return True

    # Helper methods
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

    def _to_json(self, obj):
        """Safely convert object to JSON string."""
        try:
            import json

            return json.dumps(obj)
        except:
            return str(obj)

    def _convert_tools(self, tools):
        """Convert Anthropic tools to OpenAI function format with schema cleaning."""
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

            # Clean schema for Gemini
            input_schema = self._clean_gemini_schema(tool_dict.get("input_schema", {}))

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema,
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    def _clean_gemini_schema(self, schema: Any) -> Any:
        """Recursively removes unsupported fields from a JSON schema for Gemini."""
        if isinstance(schema, dict):
            # Remove specific keys unsupported by Gemini tool parameters
            schema.pop("additionalProperties", None)
            schema.pop("default", None)

            # Check for unsupported 'format' in string types
            if schema.get("type") == "string" and "format" in schema:
                import json

                allowed_formats = {"enum", "date-time"}
                if schema["format"] not in allowed_formats:
                    logger.debug(
                        f"Removing unsupported format '{schema['format']}' for string type in Gemini schema."
                    )
                    schema.pop("format")

            # Recursively clean nested schemas
            for key in list(schema.keys()):
                schema[key] = self._clean_gemini_schema(schema[key])
        elif isinstance(schema, list):
            return [self._clean_gemini_schema(item) for item in schema]
        return schema

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
