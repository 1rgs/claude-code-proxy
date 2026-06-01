from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# ---------- 模型映射核心逻辑（合并改进） ----------
def apply_model_mapping(original_model: str, values: dict, is_token_count: bool = False) -> str:
    """
    根据环境变量和模型名称返回映射后的模型名（可能带 provider 前缀）。
    同时将原始模型名存储到 values['original_model'] 中。
    合并自 <new> 版本，但保留 <oringe> 中 PREFERRED_PROVIDER == "anthropic" 时添加前缀的行为。
    """
    logger.debug(f"📋 MODEL MAPPING: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")
    
    # 存储原始模型名
    if isinstance(values, dict):
        values['original_model'] = original_model
    
    # 移除已有的 provider 前缀，获取纯模型名
    clean_v = original_model
    if clean_v.startswith('anthropic/'):
        clean_v = clean_v[10:]
    elif clean_v.startswith('openai/'):
        clean_v = clean_v[7:]
    elif clean_v.startswith('gemini/'):
        clean_v = clean_v[7:]
    
    # 情况1：首选提供商是 anthropic → 强制转换为 anthropic/ 前缀，不进行大小模型映射
    if PREFERRED_PROVIDER == "anthropic":
        # 如果已经有 anthropic/ 前缀，直接返回原值；否则添加前缀
        if original_model.startswith('anthropic/'):
            new_model = original_model
        else:
            new_model = f"anthropic/{clean_v}"
        log_type = "TOKEN COUNT MAPPING" if is_token_count else "MODEL MAPPING"
        logger.debug(f"📌 {log_type}: '{original_model}' ➡️ '{new_model}' (forced anthropic)")
        return new_model
    
    # 情况2：其他首选提供商（openai/google） → 应用映射规则
    mapped = False
    new_model = original_model  # 默认不变
    
    # 1. Haiku -> SMALL_MODEL
    if 'haiku' in clean_v.lower():
        if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            new_model = f"gemini/{SMALL_MODEL}"
        else:
            new_model = f"openai/{SMALL_MODEL}"
        mapped = True
    
    # 2. Sonnet 或 Opus -> BIG_MODEL
    elif any(keyword in clean_v.lower() for keyword in ['sonnet', 'opus']):
        if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            new_model = f"gemini/{BIG_MODEL}"
        else:
            new_model = f"openai/{BIG_MODEL}"
        mapped = True
    
    # 3. 已知的 Gemini/OpenAI 模型，直接加上对应的前缀
    elif not mapped:
        if clean_v in GEMINI_MODELS and not original_model.startswith('gemini/'):
            new_model = f"gemini/{clean_v}"
            mapped = True
        elif clean_v in OPENAI_MODELS and not original_model.startswith('openai/'):
            new_model = f"openai/{clean_v}"
            mapped = True
    
    # 4. 如果仍未映射且无前缀，记录警告（保持原样）
    if not mapped:
        if not original_model.startswith(('openai/', 'gemini/', 'anthropic/')):
            logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
        new_model = original_model
    
    if mapped:
        log_type = "TOKEN COUNT MAPPING" if is_token_count else "MODEL MAPPING"
        logger.debug(f"📌 {log_type}: '{original_model}' ➡️ '{new_model}'")
    
    return new_model

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["user", "assistant"]
    # Keep content flexible: Claude Code may send text, tool_use/tool_result,
    # image, thinking/redacted_thinking, and future Anthropic blocks.
    content: Union[str, List[Any]]

class Tool(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Support both older {enabled: true} and current Anthropic {type, budget_tokens}.
    enabled: Optional[bool] = None
    type: Optional[str] = None
    budget_tokens: Optional[int] = None

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info):
        values = info.data
        new_model = apply_model_mapping(v, values, is_token_count=False)
        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info):
        values = info.data
        new_model = apply_model_mapping(v, values, is_token_count=True)
        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Dict[str, Any]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
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
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Return a plain dict for Pydantic models, LiteLLM objects, and dicts."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if hasattr(obj, "dict"):
        return obj.dict(exclude_none=True)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tool_arguments_to_string(value: Any) -> str:
    if value is None or value == "":
        return "{}"
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def _tool_arguments_to_object(value: Any) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": value}
    return {"value": value}


def _anthropic_tool_use_to_openai_tool_call(block: Any) -> Dict[str, Any]:
    block_dict = _as_dict(block)
    tool_id = str(block_dict.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
    name = str(block_dict.get("name") or "")
    arguments = _tool_arguments_to_string(block_dict.get("input", {}))
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _anthropic_image_to_openai_part(block: Any) -> Dict[str, Any]:
    source = _as_dict(_get(block, "source", {}))
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    if source_type == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return {"type": "text", "text": "[Unsupported image block]"}


def _flush_user_parts(messages: List[Dict[str, Any]], parts: List[Dict[str, Any]]) -> None:
    if not parts:
        return
    if len(parts) == 1 and parts[0].get("type") == "text":
        messages.append({"role": "user", "content": parts[0].get("text", "")})
    else:
        messages.append({"role": "user", "content": list(parts)})
    parts.clear()


def _normalize_openai_content(content: Any) -> Any:
    """Keep OpenAI-compatible text/image content parts; stringify only unknown blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized = []
        for part in content:
            part_dict = _as_dict(part)
            part_type = part_dict.get("type")
            if part_type in {"text", "image_url", "input_audio", "file"}:
                normalized.append(part_dict)
            elif "text" in part_dict:
                normalized.append({"type": "text", "text": str(part_dict.get("text", ""))})
            else:
                normalized.append({"type": "text", "text": _json_dumps(part_dict)})
        if len(normalized) == 1 and normalized[0].get("type") == "text":
            return normalized[0].get("text", "")
        return normalized
    return str(content)


def _normalize_tool_result_for_openai(content: Any, is_error: Optional[bool] = None) -> str:
    text = parse_tool_result_content(content)
    if is_error:
        return f"[Tool error]\n{text}" if text else "[Tool error]"
    return text


def _sanitize_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop only unsupported keys; preserve role=tool, tool_call_id and assistant.tool_calls."""
    allowed = {
        "system": {"role", "content", "name"},
        "developer": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {"role", "content", "name", "tool_calls", "function_call"},
        "tool": {"role", "content", "tool_call_id"},
        "function": {"role", "content", "name"},
    }
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        keep = allowed.get(role, {"role", "content"})
        clean = {k: v for k, v in msg.items() if k in keep}
        if role == "assistant":
            has_tool_calls = bool(clean.get("tool_calls"))
            if has_tool_calls:
                # OpenAI allows assistant.content to be null when tool_calls are present.
                if clean.get("content") in ("", [], {}):
                    clean["content"] = None
                else:
                    clean["content"] = _normalize_openai_content(clean.get("content"))
            else:
                clean["content"] = _normalize_openai_content(clean.get("content", ""))
        elif role == "tool":
            clean["tool_call_id"] = str(clean.get("tool_call_id", ""))
            clean["content"] = _normalize_tool_result_for_openai(clean.get("content", ""))
        else:
            clean["content"] = _normalize_openai_content(clean.get("content", ""))
        sanitized.append(clean)
    return sanitized


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic Messages format to LiteLLM/OpenAI chat format.

    Claude Code sends Anthropic content blocks.  For OpenAI-compatible
    backends, assistant tool_use blocks must become assistant.tool_calls, and
    user tool_result blocks must become role="tool" messages with the same
    tool_call_id.  Converting them to plain user text loses the pairing and is
    the main source of broken multi-turn Claude Code tool loops.
    """
    messages: List[Dict[str, Any]] = []

    # System prompt.
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            system_text_parts: List[str] = []
            for block in anthropic_request.system:
                block_dict = _as_dict(block)
                if block_dict.get("type") == "text" or "text" in block_dict:
                    system_text_parts.append(str(block_dict.get("text", "")))
            if system_text_parts:
                messages.append({"role": "system", "content": "\n\n".join(system_text_parts).strip()})

    # Conversation messages.
    for msg in anthropic_request.messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for block in content:
                block_type = _get(block, "type")
                if block_type == "text":
                    text_parts.append(str(_get(block, "text", "")))
                elif block_type == "tool_use":
                    tool_calls.append(_anthropic_tool_use_to_openai_tool_call(block))
                elif block_type in {"thinking", "redacted_thinking"}:
                    # Do not forward Anthropic thinking blocks to non-Anthropic providers.
                    continue
                else:
                    # Preserve unknown assistant blocks as compact text rather than dropping context.
                    block_dict = _as_dict(block)
                    if block_dict:
                        text_parts.append(_json_dumps(block_dict))

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            text = "\n".join(part for part in text_parts if part).strip()
            if tool_calls:
                assistant_msg["content"] = text or None
                assistant_msg["tool_calls"] = tool_calls
            else:
                assistant_msg["content"] = text
            messages.append(assistant_msg)
            continue

        # User messages may contain text/image blocks and tool_result blocks.
        user_parts: List[Dict[str, Any]] = []
        for block in content:
            block_type = _get(block, "type")
            if block_type == "text":
                user_parts.append({"type": "text", "text": str(_get(block, "text", ""))})
            elif block_type == "image":
                user_parts.append(_anthropic_image_to_openai_part(block))
            elif block_type == "tool_result":
                _flush_user_parts(messages, user_parts)
                tool_use_id = str(_get(block, "tool_use_id", ""))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": _normalize_tool_result_for_openai(
                        _get(block, "content", ""),
                        bool(_get(block, "is_error", False)),
                    ),
                })
            else:
                block_dict = _as_dict(block)
                if block_dict:
                    user_parts.append({"type": "text", "text": _json_dumps(block_dict)})
        _flush_user_parts(messages, user_parts)

    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith(("openai/", "gemini/")):
        max_tokens = min(max_tokens, 16384)
        logger.debug(
            f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})"
        )

    litellm_request: Dict[str, Any] = {
        "model": anthropic_request.model,
        "messages": _sanitize_openai_messages(messages),
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    if anthropic_request.thinking and anthropic_request.model.startswith("anthropic/"):
        thinking = _as_dict(anthropic_request.thinking)
        # Older local configs used {enabled: bool}; Anthropic expects {type: ...}.
        if "enabled" in thinking and "type" not in thinking:
            thinking = {"type": "enabled" if thinking.get("enabled") else "disabled"}
        litellm_request["thinking"] = thinking

    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p is not None:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k is not None:
        litellm_request["top_k"] = anthropic_request.top_k

    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")
        for tool in anthropic_request.tools:
            tool_dict = _as_dict(tool)
            if not tool_dict.get("name"):
                logger.error(f"Skipping tool without name: {tool}")
                continue
            input_schema = tool_dict.get("input_schema") or {"type": "object", "properties": {}}
            if is_gemini_model:
                input_schema = clean_gemini_schema(json.loads(_json_dumps(input_schema)))
            function_def = {
                "name": tool_dict["name"],
                "description": tool_dict.get("description", "") or "",
                "parameters": input_schema,
            }
            # Preserve Anthropic strict schema intent if callers include it.
            if "strict" in tool_dict:
                function_def["strict"] = bool(tool_dict["strict"])
            openai_tools.append({"type": "function", "function": function_def})
        if openai_tools:
            litellm_request["tools"] = openai_tools

    if anthropic_request.tool_choice:
        tool_choice_dict = _as_dict(anthropic_request.tool_choice)
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "none":
            litellm_request["tool_choice"] = "none"
        elif choice_type == "any":
            # OpenAI equivalent of Anthropic's “must call one of the tools”.
            litellm_request["tool_choice"] = "required"
        elif choice_type == "tool" and tool_choice_dict.get("name"):
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]},
            }
        else:
            litellm_request["tool_choice"] = "auto"

    return litellm_request


def _extract_message_choice(response: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if isinstance(response, dict):
        response_dict = response
    elif hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "dict"):
        response_dict = response.dict()
    else:
        response_dict = _as_dict(response)

    choices = response_dict.get("choices") or []
    choice = choices[0] if choices else {}
    if not isinstance(choice, dict):
        choice = _as_dict(choice)
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = _as_dict(message)
    usage = response_dict.get("usage") or {}
    if not isinstance(usage, dict):
        usage = _as_dict(usage)
    return {
        "id": response_dict.get("id", f"msg_{uuid.uuid4()}"),
        "message": message,
        "finish_reason": choice.get("finish_reason", "stop"),
        "usage": usage,
    }


def _openai_content_to_anthropic_text_blocks(content_value: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if content_value is None or content_value == "":
        return blocks
    if isinstance(content_value, str):
        blocks.append({"type": "text", "text": content_value})
        return blocks
    if isinstance(content_value, list):
        text_parts: List[str] = []
        for part in content_value:
            part_dict = _as_dict(part)
            if part_dict.get("type") == "text":
                text_parts.append(str(part_dict.get("text", "")))
            elif "text" in part_dict:
                text_parts.append(str(part_dict.get("text", "")))
            elif part_dict:
                text_parts.append(_json_dumps(part_dict))
        text = "\n".join(p for p in text_parts if p)
        if text:
            blocks.append({"type": "text", "text": text})
        return blocks
    blocks.append({"type": "text", "text": str(content_value)})
    return blocks


def _openai_tool_call_to_anthropic_block(tool_call: Any) -> Dict[str, Any]:
    tc = _as_dict(tool_call)
    function = tc.get("function") or {}
    if not isinstance(function, dict):
        function = _as_dict(function)
    tool_id = str(tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
    name = str(function.get("name") or tc.get("name") or "")
    arguments = function.get("arguments", "{}")
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": _tool_arguments_to_object(arguments),
    }


def _usage_tokens(usage: Dict[str, Any], prompt_key: str, completion_key: str) -> tuple[int, int]:
    prompt_tokens = usage.get(prompt_key, usage.get("input_tokens", 0)) or 0
    completion_tokens = usage.get(completion_key, usage.get("output_tokens", 0)) or 0
    try:
        prompt_tokens = int(prompt_tokens)
    except Exception:
        prompt_tokens = 0
    try:
        completion_tokens = int(completion_tokens)
    except Exception:
        completion_tokens = 0
    return prompt_tokens, completion_tokens


def _map_finish_reason(finish_reason: Optional[str], has_tool_calls: bool = False) -> str:
    if has_tool_calls or finish_reason in {"tool_calls", "function_call"}:
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "stop_sequence":
        return "stop_sequence"
    return "end_turn"


def _response_model_name(original_request: MessagesRequest) -> str:
    return original_request.original_model or original_request.model


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any],
    original_request: MessagesRequest,
) -> MessagesResponse:
    """Convert a LiteLLM/OpenAI-format response back to Anthropic Messages."""
    try:
        extracted = _extract_message_choice(litellm_response)
        message = extracted["message"]
        response_id = extracted["id"]
        finish_reason = extracted["finish_reason"]
        usage_info = extracted["usage"]

        content: List[Dict[str, Any]] = []
        content.extend(_openai_content_to_anthropic_text_blocks(message.get("content")))

        tool_calls = message.get("tool_calls") or []
        if tool_calls and not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        for tool_call in tool_calls:
            block = _openai_tool_call_to_anthropic_block(tool_call)
            logger.debug(
                f"Adding tool_use block: id={block['id']}, name={block['name']}, input={block['input']}"
            )
            content.append(block)

        if not content:
            content.append({"type": "text", "text": ""})

        prompt_tokens, completion_tokens = _usage_tokens(
            usage_info, "prompt_tokens", "completion_tokens"
        )
        stop_reason = _map_finish_reason(finish_reason, has_tool_calls=bool(tool_calls))

        return MessagesResponse(
            id=response_id,
            model=_response_model_name(original_request),
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}")
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=_response_model_name(original_request),
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _first_choice_from_chunk(chunk: Any) -> Dict[str, Any]:
    choices = _get(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not choices:
        return {}
    choice = choices[0]
    return choice if isinstance(choice, dict) else _as_dict(choice)


def _delta_from_choice(choice: Dict[str, Any]) -> Dict[str, Any]:
    delta = choice.get("delta") or choice.get("message") or {}
    return delta if isinstance(delta, dict) else _as_dict(delta)


def _finish_reason_from_choice(choice: Dict[str, Any]) -> Optional[str]:
    return choice.get("finish_reason")


def _stream_usage_from_chunk(chunk: Any) -> tuple[Optional[int], Optional[int]]:
    usage = _get(chunk, "usage", None)
    if usage is None:
        return None, None
    usage_dict = usage if isinstance(usage, dict) else _as_dict(usage)
    in_tok = usage_dict.get("prompt_tokens", usage_dict.get("input_tokens"))
    out_tok = usage_dict.get("completion_tokens", usage_dict.get("output_tokens"))
    try:
        in_tok = int(in_tok) if in_tok is not None else None
    except Exception:
        in_tok = None
    try:
        out_tok = int(out_tok) if out_tok is not None else None
    except Exception:
        out_tok = None
    return in_tok, out_tok


async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Convert LiteLLM/OpenAI streaming chunks into Anthropic SSE events.

    Important correctness points for Claude Code:
    - Do not pre-open an empty text block before a tool-only response.
    - Map each OpenAI tool_calls[index] to exactly one Anthropic content block.
    - Stream function.arguments as input_json_delta.partial_json strings.
    - Finish with message_delta.stop_reason="tool_use" when tool calls occurred.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0
    next_content_index = 0
    current_text_index: Optional[int] = None
    any_block_started = False
    finish_reason_seen: Optional[str] = None
    saw_tool_calls = False

    # OpenAI tool-call index -> state for the Anthropic content block.
    tool_states: Dict[int, Dict[str, Any]] = {}
    tool_order: List[int] = []

    def start_text_block() -> tuple[int, str]:
        nonlocal next_content_index, current_text_index, any_block_started
        idx = next_content_index
        next_content_index += 1
        current_text_index = idx
        any_block_started = True
        return idx, _sse(
            "content_block_start",
            {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}},
        )

    def start_tool_block(openai_index: int, name: str, tool_id: str) -> str:
        nonlocal next_content_index, any_block_started
        state = tool_states[openai_index]
        idx = next_content_index
        next_content_index += 1
        state["anthropic_index"] = idx
        state["started"] = True
        state["closed"] = False
        any_block_started = True
        return _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": {},
                },
            },
        )

    try:
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": _response_model_name(original_request),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 0,
                    },
                },
            },
        )
        yield _sse("ping", {"type": "ping"})

        async for chunk in response_generator:
            in_tok, out_tok = _stream_usage_from_chunk(chunk)
            if in_tok is not None:
                input_tokens = in_tok
            if out_tok is not None:
                output_tokens = out_tok

            choice = _first_choice_from_chunk(chunk)
            if not choice:
                continue
            delta = _delta_from_choice(choice)
            finish_reason = _finish_reason_from_choice(choice)

            # Text delta.
            delta_content = delta.get("content")
            if delta_content:
                if current_text_index is None:
                    idx, event = start_text_block()
                    yield event
                else:
                    idx = current_text_index
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": str(delta_content)},
                    },
                )

            # Tool-call delta(s).
            delta_tool_calls = delta.get("tool_calls") or []
            if delta_tool_calls:
                saw_tool_calls = True
                if current_text_index is not None:
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": current_text_index},
                    )
                    current_text_index = None

                if not isinstance(delta_tool_calls, list):
                    delta_tool_calls = [delta_tool_calls]

                for tool_call in delta_tool_calls:
                    tc = tool_call if isinstance(tool_call, dict) else _as_dict(tool_call)
                    openai_index = tc.get("index", 0)
                    try:
                        openai_index = int(openai_index)
                    except Exception:
                        openai_index = 0

                    if openai_index not in tool_states:
                        tool_states[openai_index] = {
                            "id": f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": "",
                            "started": False,
                            "closed": False,
                            "pending_args": [],
                            "anthropic_index": None,
                        }
                        tool_order.append(openai_index)

                    state = tool_states[openai_index]
                    if tc.get("id"):
                        state["id"] = str(tc["id"])

                    function = tc.get("function") or {}
                    if not isinstance(function, dict):
                        function = _as_dict(function)
                    if function.get("name"):
                        state["name"] = str(function["name"])

                    raw_args = function.get("arguments")
                    arg_delta = None
                    if raw_args is not None and raw_args != "":
                        arg_delta = _tool_arguments_to_string(raw_args)

                    # Start once we have a name, or once arguments begin.  Most
                    # providers send id/name before args; the fallback prevents
                    # losing args from less strict OpenAI-compatible servers.
                    if not state["started"] and (state["name"] or arg_delta is not None):
                        yield start_tool_block(
                            openai_index,
                            state["name"] or "unknown_tool",
                            state["id"],
                        )
                        for pending in state["pending_args"]:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": state["anthropic_index"],
                                    "delta": {"type": "input_json_delta", "partial_json": pending},
                                },
                            )
                        state["pending_args"].clear()

                    if arg_delta is not None:
                        if state["started"]:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": state["anthropic_index"],
                                    "delta": {"type": "input_json_delta", "partial_json": arg_delta},
                                },
                            )
                        else:
                            state["pending_args"].append(arg_delta)

            if finish_reason:
                finish_reason_seen = finish_reason
                break

        # Close any open content blocks.
        if current_text_index is not None:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": current_text_index})
            current_text_index = None

        for openai_index in tool_order:
            state = tool_states[openai_index]
            if not state["started"]:
                # Extremely defensive: close over a malformed provider stream that
                # announced a tool_call but never emitted a function name/args.
                yield start_tool_block(openai_index, state["name"] or "unknown_tool", state["id"])
            if not state.get("closed"):
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": state["anthropic_index"]},
                )
                state["closed"] = True

        if not any_block_started:
            idx, event = start_text_block()
            yield event
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})

        stop_reason = _map_finish_reason(finish_reason_seen, has_tool_calls=saw_tool_calls)
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})

    except Exception as e:
        import traceback
        logger.error(f"Error in streaming: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}")
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": f"Error in streaming conversion: {str(e)}"},
            },
        )

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        # print the body here
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        request.original_model = original_model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Determine which API key to use based on the model
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            # Use custom OpenAI base URL if configured
            if OPENAI_BASE_URL:
                litellm_request["api_base"] = OPENAI_BASE_URL
                logger.debug(f"Using OpenAI API key and custom base URL {OPENAI_BASE_URL} for model: {request.model}")
            else:
                logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            if USE_VERTEX_AUTH:
                litellm_request["vertex_project"] = VERTEX_PROJECT
                litellm_request["vertex_location"] = VERTEX_LOCATION
                litellm_request["custom_llm_provider"] = "vertex_ai"
                logger.debug(f"Using Gemini ADC with project={VERTEX_PROJECT}, location={VERTEX_LOCATION} and model: {request.model}")
            else:
                litellm_request["api_key"] = GEMINI_API_KEY
                logger.debug(f"Using Gemini API key for model: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")
        
        # Final safety pass: keep valid OpenAI/LiteLLM message shapes,
        # especially assistant.tool_calls and role="tool" results.
        # The previous implementation flattened Anthropic tool_result/tool_use
        # blocks into plain text, which breaks Claude Code's tool loop.
        if "messages" in litellm_request:
            litellm_request["messages"] = _sanitize_openai_messages(litellm_request["messages"])
            for i, msg in enumerate(litellm_request["messages"]):
                logger.debug(
                    f"Message {i} format check - role: {msg.get('role')}, "
                    f"content type: {type(msg.get('content'))}, "
                    f"tool_calls: {bool(msg.get('tool_calls'))}, "
                    f"tool_call_id: {msg.get('tool_call_id')}"
                )

        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")
        
        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # Helper function to safely serialize objects for JSON
        def sanitize_for_json(obj):
            """递归地清理对象使其可以JSON序列化"""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Log all error details with safe serialization
        sanitized_details = sanitize_for_json(error_details)
        logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Prepare token counter arguments
            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }
            
            # Add custom base URL for OpenAI models if configured
            if request.model.startswith("openai/") and OPENAI_BASE_URL:
                token_counter_args["api_base"] = OPENAI_BASE_URL
            
            # Count tokens
            token_count = token_counter(**token_counter_args)
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
