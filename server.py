from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
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
import hashlib
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format="%(asctime)s - %(levelname)s - %(message)s",
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
            "cost_calculator",
        ]

        if hasattr(record, "msg") and isinstance(record.msg, str):
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
        if record.levelno == logging.DEBUG and "MODEL MAPPING" in str(record.msg):
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)


# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(
            ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )

app = FastAPI()

# Build info for debugging deployment/runtime
BUILD_ID = datetime.now().isoformat()
FILE_PATH = __file__
try:
    FILE_HASH = hashlib.md5(open(__file__, "rb").read()).hexdigest()[:8]
except Exception:
    FILE_HASH = "unknown"

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-5")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-5-mini")

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
    "gpt-4.1-mini",  # Added default small model
    "gpt-5",
    "gpt-5-mini",
]

# List of Gemini models
GEMINI_MODELS = ["gemini-2.5-pro-preview-03-25", "gemini-2.0-flash"]


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
                logger.debug(
                    f"Removing unsupported format '{schema['format']}' for string type in Gemini schema."
                )
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(
            schema.items()
        ):  # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema


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
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: Optional[bool] = None


class MessagesRequest(BaseModel):
    model: str
    max_tokens: Optional[int] = 1024  # keep for compatibility
    max_completion_tokens: Optional[int] = None  # o1/o3 style
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

    @field_validator("model")
    def validate_model_field(cls, v, info):  # Renamed to avoid conflict
        original_model = v
        new_model = v  # Default to original value

        logger.debug(
            f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'"
        )

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith("anthropic/"):
            clean_v = clean_v[10:]
        elif clean_v.startswith("openai/"):
            clean_v = clean_v[7:]
        elif clean_v.startswith("gemini/"):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Haiku to SMALL_MODEL based on provider preference
        if "haiku" in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif "sonnet" in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith("gemini/"):
                new_model = f"gemini/{clean_v}"
                mapped = True  # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith("openai/"):
                new_model = f"openai/{clean_v}"
                mapped = True  # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
            # If no mapping occurred and no prefix exists, log warning or decide default
            if not v.startswith(("openai/", "gemini/", "anthropic/")):
                logger.warning(
                    f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is."
                )
            new_model = v  # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model

        return new_model


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator("model")
    def validate_model_token_count(cls, v, info):  # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        original_model = v
        new_model = v  # Default to original value

        logger.debug(
            f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'"
        )

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith("anthropic/"):
            clean_v = clean_v[10:]
        elif clean_v.startswith("openai/"):
            clean_v = clean_v[7:]
        elif clean_v.startswith("gemini/"):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        if "haiku" in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        elif "sonnet" in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith("gemini/"):
                new_model = f"gemini/{clean_v}"
                mapped = True
            elif clean_v in OPENAI_MODELS and not v.startswith("openai/"):
                new_model = f"openai/{clean_v}"
                mapped = True
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
            if not v.startswith(("openai/", "gemini/", "anthropic/")):
                logger.warning(
                    f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is."
                )
            new_model = v

        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model

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
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


def is_openai_reasoning_model(model: str) -> bool:
    # Works with or without provider prefix
    name = model.split("/", 1)[1].lower() if "/" in model else model.lower()
    return name.startswith(("o1", "o3"))


def effective_max_tokens(req: MessagesRequest) -> int:
    if getattr(req, "max_completion_tokens", None) is not None:
        return int(req.max_completion_tokens)
    return int(req.max_tokens or 1024)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path

    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")

    # Process the request and get the response
    response = await call_next(request)
    # Add build header for debugging
    response.headers["X-Server-Build"] = f"{FILE_HASH}-{BUILD_ID}"
    return response


@app.get("/_debug/info")
async def debug_info():
    return {
        "build_id": BUILD_ID,
        "file": FILE_PATH,
        "file_hash": FILE_HASH,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "python": sys.executable,
        "venv": os.environ.get("VIRTUAL_ENV"),
        "litellm_version": getattr(litellm, "__version__", "unknown"),
    }


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


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
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

    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            if msg.role == "user" and any(
                block.type == "tool_result"
                for block in content
                if hasattr(block, "type")
            ):
                text_content = ""
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            tool_id = (
                                block.tool_use_id
                                if hasattr(block, "tool_use_id")
                                else ""
                            )
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    for content_block in block.content:
                                        if (
                                            hasattr(content_block, "type")
                                            and content_block.type == "text"
                                        ):
                                            result_content += content_block.text + "\n"
                                        elif (
                                            isinstance(content_block, dict)
                                            and content_block.get("type") == "text"
                                        ):
                                            result_content += (
                                                content_block.get("text", "") + "\n"
                                            )
                                        elif isinstance(content_block, dict):
                                            if "text" in content_block:
                                                result_content += (
                                                    content_block.get("text", "") + "\n"
                                                )
                                            else:
                                                try:
                                                    result_content += (
                                                        json.dumps(content_block) + "\n"
                                                    )
                                                except:
                                                    result_content += (
                                                        str(content_block) + "\n"
                                                    )
                                elif isinstance(block.content, dict):
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "Unparseable content"
                            text_content += (
                                f"Tool result for {tool_id}:\n{result_content}\n"
                            )
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append(
                                {"type": "text", "text": block.text}
                            )
                        elif block.type == "image":
                            processed_content.append(
                                {"type": "image", "source": block.source}
                            )
                        elif block.type == "tool_use":
                            processed_content.append(
                                {
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                }
                            )
                        elif block.type == "tool_result":
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": (
                                    block.tool_use_id
                                    if hasattr(block, "tool_use_id")
                                    else ""
                                ),
                            }
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    processed_content_block["content"] = [
                                        {"type": "text", "text": block.content}
                                    ]
                                elif isinstance(block.content, list):
                                    processed_content_block["content"] = block.content
                                else:
                                    processed_content_block["content"] = [
                                        {"type": "text", "text": str(block.content)}
                                    ]
                            else:
                                processed_content_block["content"] = [
                                    {"type": "text", "text": ""}
                                ]
                            processed_content.append(processed_content_block)

                messages.append({"role": msg.role, "content": processed_content})

    # Compute effective max tokens and cap for OpenAI/Gemini
    eff_max = effective_max_tokens(anthropic_request)
    if anthropic_request.model.startswith(("openai/", "gemini/")):
        eff_max = min(eff_max, 16384)
        logger.debug(
            f"Capping max tokens to 16384 for OpenAI/Gemini model (original value: {effective_max_tokens(anthropic_request)})"
        )

    # Create LiteLLM request dict; add model-conditional params below
    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "stream": anthropic_request.stream,
    }

    # Add optional parameters if present, but avoid unsupported ones for o1/o3
    is_o_model = anthropic_request.model.startswith(
        "openai/"
    ) and is_openai_reasoning_model(anthropic_request.model)

    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences

    if not is_o_model:
        if anthropic_request.top_p is not None:
            litellm_request["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            litellm_request["top_k"] = anthropic_request.top_k

    # Tokens + temperature handling per model family
    if is_o_model:
        litellm_request["max_completion_tokens"] = eff_max
        # ensure o-models do not get unsupported params
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
        ):
            litellm_request.pop(k, None)
    else:
        litellm_request["max_tokens"] = eff_max
        litellm_request["temperature"] = anthropic_request.temperature

    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            if hasattr(tool, "dict"):
                tool_dict = tool.dict()
            else:
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                    logger.error(f"Could not convert tool to dict: {tool}")
                    continue

            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                logger.debug(
                    f"Cleaning schema for Gemini tool: {tool_dict.get('name')}"
                )
                input_schema = clean_gemini_schema(input_schema)

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema,
                },
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools

    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, "dict"):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice

        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]},
            }
        else:
            litellm_request["tool_choice"] = "auto"

    return litellm_request


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest
) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    try:
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

        is_claude_model = clean_model.startswith("claude-")

        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = (
                message.content if message and hasattr(message, "content") else ""
            )
            tool_calls = (
                message.tool_calls
                if message and hasattr(message, "tool_calls")
                else None
            )
            finish_reason = (
                choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            )
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            try:
                response_dict = (
                    litellm_response
                    if isinstance(litellm_response, dict)
                    else litellm_response.dict()
                )
            except AttributeError:
                try:
                    response_dict = (
                        litellm_response.model_dump()
                        if hasattr(litellm_response, "model_dump")
                        else litellm_response.__dict__
                    )
                except AttributeError:
                    response_dict = {
                        "id": getattr(litellm_response, "id", f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, "choices", [{}]),
                        "usage": getattr(litellm_response, "usage", {}),
                    }

            choices = response_dict.get("choices", [{}])
            message = (
                choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            )
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

        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        if tool_calls and is_claude_model:
            logger.debug(f"Processing tool calls: {tool_calls}")

            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

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

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool arguments as JSON: {arguments}"
                        )
                        arguments = {"raw": arguments}

                logger.debug(
                    f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}"
                )

                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": arguments,
                    }
                )
        elif tool_calls and not is_claude_model:
            logger.debug(
                f"Converting tool calls to text for non-Claude model: {clean_model}"
            )
            tool_text = "\n\nTool usage:\n"

            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
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

                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                        arguments_str = json.dumps(args_dict, indent=2)
                    except json.JSONDecodeError:
                        arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, indent=2)

                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"

            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})

        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        if not content:
            content.append({"type": "text", "text": ""})

        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

        return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        )
        logger.error(error_message)

        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": f"Error converting response: {str(e)}. Please check server logs.",
                }
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        message_data = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": original_request.model,
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
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"

        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""
        text_sent = False
        text_block_closed = False
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        async for chunk in response_generator:
            try:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    if hasattr(chunk.usage, "prompt_tokens"):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, "completion_tokens"):
                        output_tokens = chunk.usage.completion_tokens

                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    if hasattr(choice, "delta"):
                        delta = choice.delta
                    else:
                        delta = getattr(choice, "message", {})

                    finish_reason = getattr(choice, "finish_reason", None)

                    delta_content = None

                    if hasattr(delta, "content"):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and "content" in delta:
                        delta_content = delta["content"]

                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

                    delta_tool_calls = None
                    if hasattr(delta, "tool_calls"):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and "tool_calls" in delta:
                        delta_tool_calls = delta["tool_calls"]

                    if delta_tool_calls:
                        if tool_index is None:
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            elif (
                                accumulated_text
                                and not text_sent
                                and not text_block_closed
                            ):
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        for tool_call in delta_tool_calls:
                            current_index = None
                            if isinstance(tool_call, dict) and "index" in tool_call:
                                current_index = tool_call["index"]
                            elif hasattr(tool_call, "index"):
                                current_index = tool_call.index
                            else:
                                current_index = 0

                            if tool_index is None or current_index != tool_index:
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index

                                if isinstance(tool_call, dict):
                                    function = tool_call.get("function", {})
                                    name = (
                                        function.get("name", "")
                                        if isinstance(function, dict)
                                        else ""
                                    )
                                    tool_id = tool_call.get(
                                        "id", f"toolu_{uuid.uuid4().hex[:24]}"
                                    )
                                else:
                                    function = getattr(tool_call, "function", None)
                                    name = (
                                        getattr(function, "name", "")
                                        if function
                                        else ""
                                    )
                                    tool_id = getattr(
                                        tool_call,
                                        "id",
                                        f"toolu_{uuid.uuid4().hex[:24]}",
                                    )

                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""

                            arguments = None
                            if isinstance(tool_call, dict) and "function" in tool_call:
                                function = tool_call.get("function", {})
                                arguments = (
                                    function.get("arguments", "")
                                    if isinstance(function, dict)
                                    else ""
                                )
                            elif hasattr(tool_call, "function"):
                                function = getattr(tool_call, "function", None)
                                arguments = (
                                    getattr(function, "arguments", "")
                                    if function
                                    else ""
                                )

                            if arguments:
                                try:
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    args_json = arguments

                                tool_content += (
                                    args_json if isinstance(args_json, str) else ""
                                )

                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"

                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True

                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"

                        usage = {"output_tokens": output_tokens}

                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"

                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        if not has_sent_stop_reason:
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            usage = {"output_tokens": output_tokens}

            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"

            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        )
        logger.error(error_message)

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"

        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        yield "data: [DONE]\n\n"


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")

        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

        logger.debug(
            f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}"
        )

        litellm_request = convert_anthropic_to_litellm(request)

        # Determine which API key to use
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            litellm_request["api_key"] = GEMINI_API_KEY
            logger.debug(f"Using Gemini API key for model: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")

        # Extra safety cleanup for OpenAI o-models/gpt-5
        model_name = (
            request.model.split("/", 1)[1].lower()
            if "/" in request.model
            else request.model.lower()
        )
        is_openai_o_like = request.model.startswith("openai/") and (
            model_name.startswith(("o1", "o3", "gpt-5"))
        )
        if is_openai_o_like:
            # ensure we don't send unsupported params through LiteLLM
            for k in (
                "temperature",
                "top_p",
                "top_k",
                "frequency_penalty",
                "presence_penalty",
            ):
                litellm_request.pop(k, None)
            litellm_request.pop("max_tokens", None)
            if "max_completion_tokens" not in litellm_request:
                litellm_request["max_completion_tokens"] = min(
                    effective_max_tokens(request), 16384
                )

        logger.debug(
            f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}"
        )

        if request.stream:
            # Streaming path: still using LiteLLM (upgrade LiteLLM if it injects max_tokens)
            # Sanity check to catch accidental max_tokens for o-like models
            if is_openai_o_like and "max_tokens" in litellm_request:
                logger.error(
                    f"Found max_tokens in litellm_request for streaming o-model: keys={list(litellm_request.keys())}"
                )
                raise RuntimeError(
                    "max_tokens present for OpenAI o-model/gpt-5 in streaming path."
                )

            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get("model"),
                len(litellm_request["messages"]),
                num_tools,
                200,
            )
            response_generator = await litellm.acompletion(**litellm_request)

            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming path: BYPASS LiteLLM for OpenAI o1/o3/gpt-5
            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get("model"),
                len(litellm_request["messages"]),
                num_tools,
                200,
            )

            if is_openai_o_like:
                client = OpenAI(api_key=OPENAI_API_KEY)
                openai_payload = {
                    "model": request.model.split("/", 1)[1],  # strip 'openai/'
                    "messages": litellm_request["messages"],
                    "max_completion_tokens": min(effective_max_tokens(request), 16384),
                }
                if "stop" in litellm_request:
                    openai_payload["stop"] = litellm_request["stop"]

                start_time = time.time()
                resp = client.chat.completions.create(**openai_payload)
                logger.debug(
                    f"✅ RESPONSE RECEIVED (OpenAI SDK): Model={openai_payload['model']}, Time={time.time() - start_time:.2f}s"
                )
                return convert_litellm_to_anthropic(resp, request)

            # All other models use LiteLLM
            # Sanity check: avoid sending max_tokens for o-like models by mistake
            if is_openai_o_like and "max_tokens" in litellm_request:
                logger.error(
                    f"Found max_tokens in litellm_request for o-model: keys={list(litellm_request.keys())}"
                )
                raise RuntimeError(
                    "max_tokens present for OpenAI o-model/gpt-5 (non-stream). Should use max_completion_tokens."
                )

            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(
                f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s"
            )
            return convert_litellm_to_anthropic(litellm_response, request)

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()

        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback,
        }

        for attr in ["message", "status_code", "response", "llm_provider", "model"]:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        if hasattr(e, "__dict__"):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ["args", "__traceback__"]:
                    try:
                        json.dumps(value)
                        error_details[key] = value
                    except TypeError:
                        error_details[key] = str(value)

        logger.exception(
            "Error processing request: %s",
            json.dumps(error_details, indent=2, default=str),
        )

        error_message = f"Error: {str(e)}"
        if "message" in error_details and error_details["message"]:
            error_message += f"\nMessage: {error_details['message']}"
        if "response" in error_details and error_details["response"]:
            error_message += f"\nResponse: {error_details['response']}"

        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    try:
        original_model = request.original_model or request.model

        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking,
            )
        )

        try:
            from litellm import token_counter

            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get("model"),
                len(converted_request["messages"]),
                num_tools,
                200,
            )

            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )

            return TokenCountResponse(input_tokens=token_count)

        except ImportError:
            logger.error("Could not import token_counter from litellm")
            return TokenCountResponse(input_tokens=1000)

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


def log_request_beautifully(
    method, path, claude_model, openai_model, num_messages, num_tools, status_code
):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"

    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    status_str = (
        f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
        if status_code == 200
        else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    )

    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"

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
