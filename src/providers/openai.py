"""OpenAI provider implementation."""
from typing import Dict, Any, Optional, AsyncIterator, Union
import logging
import json

from .base import BaseProvider, ProviderConfig

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    def get_model_prefix(self) -> str:
        return "openai/"

    def prepare_request(self, anthropic_request: Any) -> Dict[str, Any]:
        """Convert Anthropic format to LiteLLM/OpenAI format."""
        # Import MessagesRequest type here to avoid circular import
        from ..models.request import MessagesRequest

        # Set up messages
        messages = []

        # Add system message if present
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                messages.append({"role": "system", "content": anthropic_request.system})
            elif isinstance(anthropic_request.system, list):
                system_text = ""
                for block in anthropic_request.system:
                    if hasattr(block, 'type') and block.type == "text":
                        system_text += block.text + "\n\n"
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_text += block.get("text", "") + "\n\n"
                if system_text:
                    messages.append({"role": "system", "content": system_text.strip()})

        # Add conversation messages - flatten content blocks for OpenAI
        for msg in anthropic_request.messages:
            content = msg.content
            if isinstance(content, str):
                m = {"role": msg.role, "content": content}
                messages.append(m)
            else:
                # Flatten content blocks to string for OpenAI
                text_content = ""
                for block in content:
                    block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                    if block_type == "text":
                        t = getattr(block, "text", None) or (block.get("text", "") if isinstance(block, dict) else "")
                        text_content += t + "\n"
                    elif block_type == "tool_result":
                        tool_id = getattr(block, "tool_use_id", None) or (block.get("tool_use_id", "") if isinstance(block, dict) else "")
                        bcontent = getattr(block, "content", None) if not isinstance(block, dict) else block.get("content")
                        text_content += f"Tool result for {tool_id}:\n{self._parse_tool_result_content(bcontent)}\n"
                    elif block_type == "tool_use":
                        bname = getattr(block, "name", None) or (block.get("name", "") if isinstance(block, dict) else "")
                        binput = getattr(block, "input", None) or (block.get("input", {}) if isinstance(block, dict) else {})
                        text_content += f"[Tool: {bname}]\n{self._to_json(binput)}\n\n"
                    elif block_type == "image":
                        text_content += "[Image]\n"

                m = {"role": msg.role, "content": text_content.strip() or "..."}
                messages.append(m)

        # Cap max_tokens for OpenAI models
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

        # Add tools if present
        if anthropic_request.tools:
            litellm_request["tools"] = self._convert_tools(anthropic_request.tools)
            if anthropic_request.tool_choice:
                litellm_request["tool_choice"] = self._convert_tool_choice(anthropic_request.tool_choice)

        return litellm_request

    def prepare_streaming_request(self, anthropic_request: Any) -> Dict[str, Any]:
        """Same as prepare_request but with stream=True."""
        req = self.prepare_request(anthropic_request)
        req["stream"] = True
        return req

    def convert_response(self, litellm_response: Union[Dict, Any], original_request: Any) -> Any:
        """Convert OpenAI response to Anthropic format."""
        from ..models.request import MessagesResponse, Usage
        import uuid
        import json

        # Extract response data
        response_dict = self._extract_response_dict(litellm_response)

        choices = response_dict.get("choices", [{}])
        message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
        content_text = message.get("content", "")
        tool_calls = message.get("tool_calls", None)
        finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
        usage_info = response_dict.get("usage", {})
        response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        # Build Anthropic content blocks
        content = []

        # Add text content
        if content_text and content_text != "":
            content.append({"type": "text", "text": content_text})

        # Convert tool calls to text format for non-Anthropic models
        if tool_calls:
            tool_text = "\n\nTool usage:\n"
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"

                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                        arguments_str = json.dumps(args_dict, indent=2)
                    except json.JSONDecodeError:
                        arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, indent=2)

                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"

            # Append tool text to content
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})

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
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )

    async def stream_response(self, litellm_generator: AsyncIterator) -> AsyncIterator[str]:
        """Stream in Anthropic SSE format."""
        import json
        import uuid

        # Send message_start event
        message_id = f"msg_{uuid.uuid4()}"
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant'}})}\n\n"

        # Send text block start
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        # Track state
        message_delta = {"stop_reason": None, "usage": None}
        accumulated_text = ""
        text_block_closed = False

        async for chunk in litellm_generator:
            # Extract choice and delta
            choice = None
            delta = None
            finish_reason = None

            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, 'delta', None)
                finish_reason = getattr(choice, 'finish_reason', None)
            elif isinstance(chunk, dict):
                choice = chunk.get('choices', [{}])[0] if chunk.get('choices') else {}
                delta = choice.get('delta', {})
                finish_reason = choice.get('finish_reason')

            # Extract delta content
            delta_content = None
            if delta:
                if hasattr(delta, 'content'):
                    delta_content = delta.content
                elif isinstance(delta, dict):
                    delta_content = delta.get('content')

            # Process text content
            if delta_content and delta_content != "":
                if text_block_closed:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    text_block_closed = False
                accumulated_text += delta_content
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

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

                if not text_block_closed:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    text_block_closed = True

                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': None})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                yield "[DONE]\n"
                break

        # If we didn't get a finish_reason, close the response
        if not message_delta["stop_reason"]:
            if not text_block_closed:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': None})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "[DONE]\n"

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
        """Convert Anthropic tools to OpenAI function format."""
        openai_tools = []
        for tool in tools:
            if hasattr(tool, 'dict'):
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
                    "parameters": tool_dict.get("input_schema", {})
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools

    def _convert_tool_choice(self, tool_choice):
        """Convert Anthropic tool_choice to OpenAI format."""
        if hasattr(tool_choice, 'dict'):
            tool_choice_dict = tool_choice.dict()
        else:
            tool_choice_dict = tool_choice

        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            return "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            return {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        return "auto"

    def _extract_response_dict(self, response) -> Dict[str, Any]:
        """Extract dict from various response types."""
        if isinstance(response, dict):
            return response
        if hasattr(response, 'model_dump') and callable(getattr(response, 'model_dump')):
            return response.model_dump()
        if hasattr(response, 'dict') and callable(getattr(response, 'dict')):
            return response.dict()
        return response.__dict__ if hasattr(response, '__dict__') else {}
