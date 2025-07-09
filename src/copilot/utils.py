from typing import Any, Dict


def convert_openai_to_anthropic_request(
    openai_request: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Converts an OpenAI Chat Completion request dictionary to an Anthropic Messages API request dictionary.
    """
    model = openai_request.get("model", "claude-3-opus-20240229")
    # Anthropic model names don't have the "anthropic/" prefix
    if model.startswith("anthropic/"):
        model = model.split("anthropic/")[1]

    anthropic_request = {
        "model": model,
        "max_tokens": openai_request.get("max_tokens", 4096),
        "messages": [],
    }

    # Transfer optional parameters
    if "temperature" in openai_request:
        anthropic_request["temperature"] = openai_request["temperature"]
    if "top_p" in openai_request:
        anthropic_request["top_p"] = openai_request["top_p"]
    if "stream" in openai_request:
        anthropic_request["stream"] = openai_request["stream"]
    if "stop" in openai_request:
        anthropic_request["stop_sequences"] = openai_request["stop"]

    # Convert messages and system prompt
    openai_messages = openai_request.get("messages", [])
    anthropic_messages = []
    for message in openai_messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            anthropic_request["system"] = content
            continue

        if role == "user":
            user_content = []
            if isinstance(content, str):
                user_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        user_content.append(
                            {"type": "text", "text": part.get("text", "")}
                        )
                    elif part.get("type") == "image_url":
                        image_url = part["image_url"]["url"]
                        if image_url.startswith("data:"):
                            media_type = image_url.split(":")[1].split(";")[0]
                            base64_data = image_url.split(",")[1]
                            user_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data,
                                    },
                                }
                            )
            anthropic_messages.append({"role": "user", "content": user_content})

        elif role == "assistant":
            assistant_content = []
            if message.get("content"):
                assistant_content.append({"type": "text", "text": message["content"]})
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function", {})
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id"),
                            "name": function.get("name"),
                            "input": json.loads(function.get("arguments", "{}")),
                        }
                    )
            anthropic_messages.append(
                {"role": "assistant", "content": assistant_content}
            )

        elif role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id"),
                            "content": message.get("content"),
                        }
                    ],
                }
            )

    anthropic_request["messages"] = anthropic_messages

    # Convert tools
    if "tools" in openai_request:
        anthropic_request["tools"] = []
        for tool in openai_request["tools"]:
            if tool.get("type") == "function":
                function_spec = tool.get("function", {})
                anthropic_request["tools"].append(
                    {
                        "name": function_spec.get("name"),
                        "description": function_spec.get("description"),
                        "input_schema": function_spec.get("parameters"),
                    }
                )

    return anthropic_request
