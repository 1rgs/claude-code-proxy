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

# Load environment variables from .env file
load_dotenv(override=True)

# === OpenClaw Token Extraction ===
def get_openclaw_tokens():
    tokens = {
        "anthropic": None,
        "openai": None,
        "gemini": None
    }
    try:
        auth_file = os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json")
        if os.path.exists(auth_file):
            with open(auth_file, "r") as f:
                data = json.load(f)
                profiles = data.get("profiles", {})
                
                # Extract Gemini OAuth Token (prefer gemini-cli OAuth, fallback to google:default key)
                gemini_cli_profile = next(
                    (k for k in profiles if k.startswith("google-gemini-cli:")), None
                )
                if gemini_cli_profile:
                    tokens["gemini"] = profiles[gemini_cli_profile].get("access")
                elif "google:default" in profiles:
                    tokens["gemini"] = profiles["google:default"].get("key")
                
                # Extract OpenAI Codex OAuth Token
                if "openai-codex:default" in profiles:
                    tokens["openai"] = profiles["openai-codex:default"].get("access")
                    
                # Extract Anthropic Token (prefer default OAuth, fallback to manual)
                if "anthropic:default" in profiles:
                    tokens["anthropic"] = profiles["anthropic:default"].get("token")
                elif "anthropic:manual" in profiles:
                    tokens["anthropic"] = profiles["anthropic:manual"].get("token")
                    
            print("[proxy] loaded API tokens from OpenClaw auth-profiles.json")
    except Exception as e:
        print(f"[proxy] failed to load OpenClaw tokens: {e}")
        
    return tokens

OPENCLAW_TOKENS = get_openclaw_tokens()

# Update LiteLLM environment with these tokens if they aren't already set
if OPENCLAW_TOKENS["gemini"] and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = OPENCLAW_TOKENS["gemini"]
if OPENCLAW_TOKENS["openai"] and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENCLAW_TOKENS["openai"]
# Anthropic: no longer using direct API keys — all Claude traffic goes through OpenClaw gateway
# which manages OAuth tokens (Claude Max). No ANTHROPIC_API_KEY needed.

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Production level
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

# Gateway configuration for Claude models (uses OpenAI-compatible format)
GATEWAY_URL = os.environ.get("GATEWAY_URL")  # e.g., http://127.0.0.1:18789/v1
GATEWAY_KEY = os.environ.get("GATEWAY_KEY")  # Gateway auth token

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
        schema.pop("$schema", None)

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
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True

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
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai-codex/'):
            clean_v = clean_v[13:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        is_claude_model = 'claude' in clean_v.lower() or 'opus' in clean_v.lower() or 'sonnet' in clean_v.lower() or 'haiku' in clean_v.lower()
        is_gemini_model = 'gemini' in clean_v.lower()
        is_gpt_model = 'gpt' in clean_v.lower() or 'o1' in clean_v.lower() or 'o3' in clean_v.lower()
        
        if PREFERRED_PROVIDER == "anthropic" and is_claude_model:
            # Claude models always use LiteLLM direct (Anthropic API natively supports
            # tool_use/tool_result), NOT gateway passthrough (which uses OpenAI format
            # and Anthropic rejects role="tool" messages).
            new_model = f"anthropic/{clean_v}"
            mapped = True
        
        # Gemini models: route via gateway (OpenClaw manages OAuth tokens)
        elif is_gemini_model:
            if GATEWAY_URL and GATEWAY_KEY:
                gemini_model = clean_v
                if gemini_model.startswith("google/"):
                    gemini_model = gemini_model[len("google/"):]
                new_model = f"openai/google/{gemini_model}"
            else:
                if clean_v.startswith("google/"):
                    clean_v = clean_v[len("google/"):]
                new_model = f"gemini/{clean_v}"
            mapped = True
        
        # GPT/OpenAI models use openai/ prefix
        elif is_gpt_model:
            new_model = f"openai/{clean_v}"
            mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference (fallback)
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet/Opus to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower() or 'opus' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"MODEL MAPPING: '{original_model}' -> '{new_model}'")
        else:
             # If no mapping occurred and no prefix exists, log warning or decide default
             if not v.startswith(('openai/', 'openai-codex/', 'gemini/', 'google/', 'anthropic/')):
                 logger.warning(f"No prefix or mapping rule for model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

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
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai-codex/'):
            clean_v = clean_v[13:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        is_gemini_model = 'gemini' in clean_v.lower()
        is_gpt_model = 'gpt' in clean_v.lower() or 'o1' in clean_v.lower() or 'o3' in clean_v.lower()

        # Direct Gemini/OpenAI model handling
        if is_gemini_model:
            if GATEWAY_URL and GATEWAY_KEY:
                gemini_model = clean_v
                if gemini_model.startswith("google/"):
                    gemini_model = gemini_model[len("google/"):]
                new_model = f"openai/google/{gemini_model}"
            else:
                if clean_v.startswith("google/"):
                    clean_v = clean_v[len("google/"):]
                new_model = f"gemini/{clean_v}"
            mapped = True

        elif is_gpt_model:
            new_model = f"openai/{clean_v}"
            mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet/Opus to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower() or 'opus' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"TOKEN COUNT MAPPING: '{original_model}' -> '{new_model}'")
        else:
             if not v.startswith(('openai/', 'openai-codex/', 'gemini/', 'google/', 'anthropic/')):
                 logger.warning(f"No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

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

def _rehydrate_text_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recover structured tool_use/tool_result blocks from text-serialized history.
    
    Claude Code sometimes serializes assistant tool_use and user tool_result into
    plain text like:
      Assistant: "[Tool: Bash (ID: func-123)]\nInput: {\"command\":\"ls\"}\n\nOutput text..."
      User:      "[Tool Result ID: func-123]\nls output\n"
    
    This destroys structure needed by Gemini/Claude-via-proxy for multi-turn tool calling.
    We parse these patterns back into content blocks.
    """
    # Pattern for assistant tool use blocks
    # Matches: [Tool: ToolName (ID: tool-id-123)]
    # Followed by: Input: {json}
    TOOL_USE_PATTERN = re.compile(
        r'\[Tool:\s*(\w+)\s*\(ID:\s*([^)]+)\)\]\s*'
        r'Input:\s*(\{.*?\})\s*',
        re.DOTALL
    )
    
    # Pattern for user tool result blocks  
    # Matches: [Tool Result ID: tool-id-123]
    # Followed by content until next [Tool Result or end
    TOOL_RESULT_PATTERN = re.compile(
        r'\[Tool Result ID:\s*([^\]]+)\]\s*',
    )
    
    rehydrated = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if not isinstance(content, str) or not content.strip():
            rehydrated.append(msg)
            continue
        
        if role == "assistant" and "[Tool:" in content and "(ID:" in content:
            # Parse out tool_use blocks from text
            blocks = []
            last_end = 0
            
            for match in TOOL_USE_PATTERN.finditer(content):
                # Add any text before this tool block
                pre_text = content[last_end:match.start()].strip()
                if pre_text:
                    blocks.append({"type": "text", "text": pre_text})
                
                tool_name = match.group(1)
                tool_id = match.group(2).strip()
                try:
                    tool_input = json.loads(match.group(3))
                except json.JSONDecodeError:
                    tool_input = {"raw": match.group(3)}
                
                blocks.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input
                })
                last_end = match.end()
            
            # Add any trailing text after last tool block
            trailing = content[last_end:].strip()
            if trailing:
                blocks.append({"type": "text", "text": trailing})
            
            if blocks:
                rehydrated.append({"role": "assistant", "content": blocks})
                logger.info(f"[rehydrate] Recovered {sum(1 for b in blocks if b['type']=='tool_use')} tool_use blocks from assistant text")
            else:
                rehydrated.append(msg)
                
        elif role == "user" and "[Tool Result ID:" in content:
            # Parse out tool_result blocks from text
            blocks = []
            parts = TOOL_RESULT_PATTERN.split(content)
            # parts = [pre_text, tool_id_1, result_1, tool_id_2, result_2, ...]
            
            if parts[0].strip():
                blocks.append({"type": "text", "text": parts[0].strip()})
            
            # Iterate pairs: (tool_id, result_content)
            for i in range(1, len(parts), 2):
                tool_id = parts[i].strip()
                result_text = parts[i+1].strip() if i+1 < len(parts) else ""
                
                # Strip system-reminder blocks from result text
                result_text = re.sub(r'<system-reminder>.*?</system-reminder>\s*', '', result_text, flags=re.DOTALL).strip()
                
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text or "..."
                })
            
            if blocks:
                rehydrated.append({"role": "user", "content": blocks})
                logger.info(f"[rehydrate] Recovered {sum(1 for b in blocks if b['type']=='tool_result')} tool_result blocks from user text")
            else:
                rehydrated.append(msg)
        else:
            rehydrated.append(msg)
    
    return rehydrated


# --- Gemini thought_signature cache ---
# Gemini 3.x requires thought_signature in function calls when thinking is enabled.
# We cache these from the streaming response and inject them back in subsequent requests.
THOUGHT_SIG_CACHE_FILE = "/tmp/gemini-thought-sig-cache.json"

def _cache_thought_sig(tool_call_id: str, thought_signature: str):
    """Cache a thought_signature for a tool_call_id."""
    if not tool_call_id or not thought_signature:
        return
    try:
        cache = {}
        try:
            with open(THOUGHT_SIG_CACHE_FILE, "r") as f:
                cache = json.load(f)
        except:
            pass
        cache[tool_call_id] = thought_signature
        # Keep cache small — only last 50 entries
        if len(cache) > 50:
            keys = list(cache.keys())
            for k in keys[:-50]:
                del cache[k]
        with open(THOUGHT_SIG_CACHE_FILE, "w") as f:
            json.dump(cache, f)
        logger.info(f"[thought_sig] Cached thought_signature for tool_call {tool_call_id} ({len(thought_signature)} chars)")
    except Exception as e:
        logger.error(f"[thought_sig] Cache write failed: {e}")

def _get_thought_sig(tool_call_id: str) -> str:
    """Get cached thought_signature for a tool_call_id."""
    try:
        with open(THOUGHT_SIG_CACHE_FILE, "r") as f:
            cache = json.load(f)
        return cache.get(tool_call_id, "")
    except:
        return ""

def _inject_thought_sigs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inject cached thought_signatures into assistant messages with tool_calls.

    Gemini expects: tc.extra_content.google.thought_signature
    """
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                # Skip if already has thought_signature
                existing = tc.get("extra_content", {}).get("google", {}).get("thought_signature", "")
                if tc_id and not existing:
                    sig = _get_thought_sig(tc_id)
                    if sig:
                        tc["extra_content"] = {"google": {"thought_signature": sig}}
                        logger.info(f"[thought_sig] Injected thought_signature for tool_call {tc_id}")
    return messages


def normalize_messages_for_gateway(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic-style content blocks to proper OpenAI chat/completions format.
    
    Claude Code merges multi-turn tool calls into single assistant messages and 
    tool_results into single user messages. We must perfectly RE-INTERLEAVE them 
    to satisfy OpenAI's strict tool call schema.
    """
    # Step 0: Rehydrate any text-serialized tool calls back into structured blocks
    messages = _rehydrate_text_tool_calls(messages)
    
    # 1. Extract ALL blocks across ALL messages sequentially
    all_blocks = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            if content.strip():
                all_blocks.append((role, {"type": "text", "text": content}))
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    all_blocks.append((role, b))
                else:
                    all_blocks.append((role, {"type": "text", "text": str(b)}))

    # 2. Rebuild proper OpenAI message sequence
    normalized = []
    
    # We'll group adjacent blocks of the same "turn type"
    # An assistant turn can contain text and tool_calls
    # A tool turn is a single role=tool message
    # A user turn contains text
    
    assistant_text_buffer = []
    assistant_tool_calls_buffer = []
    user_text_buffer = []
    
    def flush_assistant():
        nonlocal assistant_text_buffer, assistant_tool_calls_buffer
        if assistant_text_buffer or assistant_tool_calls_buffer:
            msg = {"role": "assistant", "content": "\n".join(assistant_text_buffer) if assistant_text_buffer else ""}
            if assistant_tool_calls_buffer:
                msg["tool_calls"] = assistant_tool_calls_buffer
            normalized.append(msg)
            assistant_text_buffer = []
            assistant_tool_calls_buffer = []
            
    def flush_user():
        nonlocal user_text_buffer
        if user_text_buffer:
            text = "\n".join(user_text_buffer).strip()
            if text:
                normalized.append({"role": "user", "content": text})
            user_text_buffer = []
            
    for role, block in all_blocks:
        btype = block.get("type", "")
        
        if role == "assistant":
            # Assistant blocks end any pending user turn
            flush_user()
            
            if btype == "text":
                t = block.get("text", "").strip()
                if t:
                    assistant_text_buffer.append(t)
            elif btype == "tool_use":
                assistant_tool_calls_buffer.append({
                    "id": block.get("id", f"call_{len(assistant_tool_calls_buffer)}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })
                
        elif role == "user":
            if btype == "tool_result":
                # A tool_result MUST follow the assistant's tool_calls
                # So we flush any pending assistant turn first!
                flush_assistant()
                flush_user() # Also flush any pending user text that came before this tool_result
                
                tool_id = block.get("tool_use_id", "")
                res = parse_tool_result_content(block.get("content", ""))
                normalized.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": res or "[Tool executed successfully with no output]"
                })
            elif btype == "text":
                # User text implies a new user turn, so flush assistant
                flush_assistant()
                
                t = block.get("text", "").strip()
                if t:
                    user_text_buffer.append(t)
    
    # Flush any remaining buffers at the very end
    flush_assistant()
    flush_user()
    
    # Edge case: If assistant buffer was totally empty but role was assistant, ensure we don't drop empty responses if they were meant to be empty
    if not normalized and messages:
        return messages
    
    # Debug: dump normalized messages (sequential)
    try:
        # Append to sequential log for debugging multi-turn
        import time as _time
        seq_file = "/tmp/proxy-normalize-seq.jsonl"
        with open(seq_file, "a") as _sf:
            _sf.write(json.dumps({"ts": _time.time(), "num_messages": len(normalized), 
                "roles": [m.get("role","?") for m in normalized],
                "has_tool_calls": any(m.get("tool_calls") for m in normalized)}) + "\n")
    except: pass
    try:
        with open("/tmp/proxy-debug-post-normalize.json", "w") as _f:
            debug_out = []
            for m in normalized:
                r = m.get("role", "?")
                entry = {"role": r}
                if m.get("tool_calls"):
                    entry["tool_calls"] = [{"id": tc.get("id"), "name": tc.get("function", {}).get("name")} for tc in m["tool_calls"]]
                if m.get("tool_call_id"):
                    entry["tool_call_id"] = m["tool_call_id"]
                c = m.get("content", "")
                if isinstance(c, str):
                    entry["content_preview"] = c[:300]
                    entry["content_len"] = len(c)
                elif c is None:
                    entry["content"] = None
                else:
                    entry["content_type"] = str(type(c))
                debug_out.append(entry)
            json.dump(debug_out, _f, indent=2, ensure_ascii=False)
    except: pass
    
    return normalized


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages (skip system messages, they're already handled)
    for idx, msg in enumerate(anthropic_request.messages):
        # Skip system role messages (Claude Code sends system as first message)
        if msg.role == "system":
            # If we don't have a system message yet, add it
            if not any(m.get("role") == "system" for m in messages):
                content = msg.content
                if isinstance(content, str):
                    messages.insert(0, {"role": "system", "content": content})
                elif isinstance(content, list):
                    # Extract text from content blocks
                    sys_text = ""
                    for block in content:
                        if hasattr(block, "text"):
                            sys_text += block.text + "\n"
                        elif isinstance(block, dict) and "text" in block:
                            sys_text += block["text"] + "\n"
                    if sys_text:
                        messages.insert(0, {"role": "system", "content": sys_text.strip()})
            continue
        
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # Preserve tool_result structure as content blocks so normalize_messages_for_gateway
                # can properly convert them to OpenAI role=tool messages with tool_call_id
                processed_blocks = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_blocks.append({"type": "text", "text": block.text})
                        elif block.type == "tool_result":
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            result_content = parse_tool_result_content(
                                block.content if hasattr(block, "content") else ""
                            )
                            processed_blocks.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_content
                            })
                
                messages.append({"role": "user", "content": processed_blocks})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Handle tool use blocks if needed
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, keep it
                                    processed_content_block["content"] = block.content
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [{"type": "text", "text": ""}]
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})")
    
    # Gemini 2.5 models use thinking by default, which consumes output tokens.
    # Enforce a minimum of 8192 so thinking doesn't eat all available output space.
    if anthropic_request.model.startswith("gemini/") and any(v in anthropic_request.model for v in ["2.5", "3-pro", "3-flash"]):
        if max_tokens < 8192:
            logger.info(f"Bumping max_tokens from {max_tokens} to 8192 for Gemini 2.5 thinking model")
            max_tokens = 8192
    
    # Extract system messages and convert to system_instruction for Gemini
    system_content = None
    non_system_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            if system_content is None:
                system_content = msg.get("content", "")
            else:
                system_content += "\n\n" + msg.get("content", "")
        else:
            non_system_messages.append(msg)
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": non_system_messages,
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    
    # Add system instruction - different handling per provider
    if system_content:
        if anthropic_request.model.startswith("gemini/"):
            # Gemini uses system parameter
            litellm_request["system"] = system_content
        elif anthropic_request.model.startswith("openai/"):
            # OpenAI/Azure models: prepend system message to messages array
            litellm_request["messages"].insert(0, {"role": "system", "content": system_content})
        else:
            # Default: use system parameter
            litellm_request["system"] = system_content

    # Include thinking field for Claude/Anthropic models
    # In /real/ routes, the model might not have the anthropic/ prefix yet,
    # but if thinking is present, we must convert it to dict to avoid LiteLLM crashes
    if anthropic_request.thinking and ('anthropic' in anthropic_request.model or 'claude' in anthropic_request.model.lower() or 'opus' in anthropic_request.model.lower()):
        # Convert Pydantic model to dict for LiteLLM compatibility
        thinking = anthropic_request.thinking
        # Anthropic API requires exactly: {"type": "enabled", "budget_tokens": N}
        is_enabled = thinking.enabled if hasattr(thinking, 'enabled') else True
        litellm_request["thinking"] = {
            "type": "enabled" if is_enabled else "disabled",
            "budget_tokens": max_tokens - 1000 if max_tokens > 2000 else max_tokens
        }

    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/") or anthropic_request.model.startswith("openai/google/")

        # DEBUG: dump tool names for analysis
        tool_names = [getattr(t, 'name', None) or (t.get('name') if isinstance(t, dict) else '?') for t in anthropic_request.tools]
        logger.debug(f"TOOLS ({len(anthropic_request.tools)}): {tool_names}")

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            # DEBUG: dump Task tool schema for analysis
            if tool_dict["name"] == "Task":
                import json as _json
                logger.debug(f"TASK TOOL SCHEMA (Anthropic): {_json.dumps(tool_dict.get('input_schema', {}), indent=2)[:1000]}")
                logger.debug(f"TASK TOOL SCHEMA (OpenAI): {_json.dumps(openai_tool, indent=2)[:1000]}")
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format) - ALL models
        # Claude CLI expects tool_use blocks regardless of which model generated them
        if tool_calls:
            logger.debug(f"Processing tool calls for model {clean_model}: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Fix: if response has tool_use blocks but finish_reason is "stop",
        # force stop_reason to "tool_use" — Gemini/GPT sometimes return "stop"
        # even when the response contains tool calls that need execution
        has_tool_use = any(c.get("type") == "tool_use" if isinstance(c, dict) else getattr(c, "type", None) == "tool_use" for c in content)
        if stop_reason == "end_turn" and has_tool_use:
            logger.warning(f"[litellm-convert] finish_reason='stop' but response has tool_use blocks — forcing stop_reason='tool_use'")
            stop_reason = "tool_use"
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        # Use original_model (before mapping) so CLI sees a Claude model name
        response_model = original_request.original_model or original_request.model
        anthropic_response = MessagesResponse(
            id=response_id,
            model=response_model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.original_model or original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.original_model or original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk
        chunk_count = 0
        logger.debug(f"Streaming START model={original_request.model}")
        async for chunk in response_generator:
            try:
                chunk_count += 1

                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments
                                
                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Fix: if response has tool_use blocks but finish_reason is "stop",
                        # force stop_reason to "tool_use"
                        if stop_reason == "end_turn" and tool_index is not None:
                            logger.warning(f"[litellm-stream] finish_reason='stop' but response has tool calls — forcing stop_reason='tool_use'")
                            stop_reason = "tool_use"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk#{chunk_count} model={original_request.model}: {str(e)}")
                continue

        logger.debug(f"Streaming DONE model={original_request.model} chunks={chunk_count} has_stop={has_sent_stop_reason} text_len={len(accumulated_text)} tool_idx={tool_index}")

        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/real/{real_model_path:path}/messages")
@app.post("/real/{real_model_path:path}/v1/messages")
async def create_message_with_override(
    request: MessagesRequest,
    raw_request: Request,
    real_model_path: str
):
    """Route for proxied requests where real model is embedded in URL path.
    Two patterns: /v1/real/<model>/messages (legacy) and /real/<model>/v1/messages (Claude CLI).
    Claude CLI appends /v1/messages to ANTHROPIC_BASE_URL, so path can be:
      /v1/real/gpt-5.3-codex/v1/messages → real_model_path = "gpt-5.3-codex/v1"
    We strip the trailing /v1 to get the actual model name."""
    # Strip trailing /v1 that Claude CLI appends
    clean_model = real_model_path
    if clean_model.endswith("/v1"):
        clean_model = clean_model[:-3]
    return await create_message(request, raw_request, real_model=clean_model)

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request,
    real_model: Optional[str] = None
):
    try:
        # print the body here
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # DEBUG: dump raw Anthropic messages for multi-turn debugging
        try:
            raw_msgs = body_json.get("messages", [])
            if len(raw_msgs) > 1:  # Multi-turn
                with open("/tmp/proxy-debug-raw-request.json", "w") as _rf:
                    debug_raw = []
                    for m in raw_msgs:
                        entry = {"role": m.get("role","?")}
                        c = m.get("content")
                        if isinstance(c, str):
                            entry["content"] = c[:200]
                        elif isinstance(c, list):
                            entry["blocks"] = [{"type": b.get("type","?"), "content": str(b.get("content","") or b.get("text","") or b.get("input",""))[:200]} for b in c if isinstance(b, dict)]
                        else:
                            entry["content"] = str(c)[:200]
                        debug_raw.append(entry)
                    json.dump(debug_raw, _rf, indent=2, ensure_ascii=False)
        except: pass
        
        # === Real model override (from ?real_model= query param) ===
        # When Claude CLI is tricked with --model opus but the real model is different,
        # persistent-session passes the real model via ANTHROPIC_BASE_URL query param.
        if real_model:
            logger.info(f"REAL MODEL OVERRIDE: '{request.model}' -> routing as '{real_model}'")
            # Re-map the model to the real one with proper prefix
            # IMPORTANT: For overridden models, always use direct API (not gateway)
            # because gateway routes to OpenClaw agent instead of the actual model
            real_lower = real_model.lower()
            if 'gemini' in real_lower:
                rm = real_model.replace("google/", "")
                request.model = f"gemini/{rm}"  # Always direct, never via gateway
            elif 'gpt' in real_lower or 'o1' in real_lower or 'o3' in real_lower or 'codex' in real_lower:
                rm = real_model.split("/")[-1]  # Strip provider prefix (openai-codex/gpt-5.3-codex → gpt-5.3-codex)
                request.model = f"openai/{rm}"
            else:
                request.model = real_model
            original_model = real_model
            logger.info(f"Model after override: {request.model}")

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
        
        logger.debug(f"PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Route decision: ALL models go through OpenClaw gateway passthrough.
        # OpenClaw manages OAuth tokens for all providers (Claude Max, Gemini, GPT).
        # Proxy only does protocol conversion (Anthropic format → OpenAI format).
        # Messages are normalized to OpenAI format before sending to gateway.
        use_gateway_direct = bool(GATEWAY_URL and GATEWAY_KEY)
        if use_gateway_direct:
            logger.info(f"Using Gateway passthrough for model: {request.model} -> {GATEWAY_URL}")
        else:
            # Fallback: Direct API calls via LiteLLM (only if gateway not configured)
            if request.model.startswith("openai/"):
                litellm_request["api_key"] = OPENAI_API_KEY
            elif request.model.startswith("gemini/"):
                litellm_request["api_key"] = GEMINI_API_KEY
            else:
                litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using LiteLLM direct (no gateway): {request.model}")
        
        # For OpenAI models only - legacy text-flattening (kept for compatibility)
        # Non-OpenAI models use normalize_messages_for_gateway instead
        if False and "messages" in litellm_request:  # DISABLED - normalize handles all models now
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message
                
                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"
                                    
                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                
                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                
                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"
                        
                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed
                
                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]
            
            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                
                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

        # --- Gateway direct path: bypass LiteLLM, use httpx with custom header ---
        if use_gateway_direct:
            import httpx
            gateway_endpoint = f"{GATEWAY_URL}/chat/completions"
            # All models use claude-code-raw passthrough — gateway handles
            # provider detection, auth (OAuth), and format conversion uniformly.
            gateway_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GATEWAY_KEY}",
                "x-openclaw-agent-id": "claude-code-raw",
            }
            # Strip litellm provider prefixes - gateway detects provider from model name
            gw_model = litellm_request["model"]
            for prefix in ["openai-codex/", "openai-relay/", "openai/", "gemini/", "anthropic/", "google/"]:
                if gw_model.startswith(prefix):
                    gw_model = gw_model[len(prefix):]
                    break
            # Normalize messages to proper OpenAI format (tool_use → tool_calls, tool_result → role=tool)
            normalized_messages = normalize_messages_for_gateway(litellm_request["messages"])
            # Inject cached thought_signatures for Gemini round-trip
            normalized_messages = _inject_thought_sigs(normalized_messages)
            gateway_body = {
                "model": gw_model,
                "messages": normalized_messages,
                "stream": bool(request.stream),
            }
            if litellm_request.get("max_tokens"):
                gateway_body["max_tokens"] = litellm_request["max_tokens"]
            if litellm_request.get("tools"):
                gateway_body["tools"] = litellm_request["tools"]
            if litellm_request.get("tool_choice"):
                gateway_body["tool_choice"] = litellm_request["tool_choice"]
            if litellm_request.get("temperature") is not None:
                gateway_body["temperature"] = litellm_request["temperature"]

            num_tools = len(request.tools) if request.tools else 0
            # Dump full request for debugging
            try:
                with open("/tmp/proxy-debug-request.json", "w") as _f:
                    # Dump messages (truncated) + tools
                    debug_msgs = []
                    for m in gateway_body.get("messages", []):
                        role = m.get("role", "?")
                        content = m.get("content", "")
                        if isinstance(content, str):
                            debug_msgs.append({"role": role, "content_preview": content[:2000], "content_len": len(content)})
                        elif isinstance(content, list):
                            blocks_info = []
                            for blk in content:
                                if isinstance(blk, dict):
                                    btype = blk.get("type", "?")
                                    btext = blk.get("text", "")
                                    blocks_info.append({"type": btype, "len": len(str(btext)), "preview": str(btext)[:500]})
                                else:
                                    blocks_info.append({"raw": str(blk)[:200]})
                            debug_msgs.append({"role": role, "content_blocks": len(content), "blocks": blocks_info})
                        else:
                            debug_msgs.append({"role": role, "content_type": str(type(content))})
                    json.dump({"model": gateway_body["model"], "num_tools": num_tools, "messages": debug_msgs}, _f, indent=2)
            except Exception as _e:
                logger.error(f"Failed to dump request: {_e}")
            if gateway_body.get("tools"):
                # Dump ALL tools to /tmp for debugging
                try:
                    with open("/tmp/proxy-debug-tools.json", "w") as _f:
                        json.dump({"model": gateway_body["model"], "num_tools": len(gateway_body["tools"]), "tools": gateway_body["tools"]}, _f, indent=2)
                    logger.info(f"Dumped {len(gateway_body['tools'])} tools to /tmp/proxy-debug-tools.json")
                except Exception as _e:
                    logger.error(f"Failed to dump tools: {_e}")
                for t in gateway_body["tools"]:
                    fname = t.get("function", {}).get("name", "")
                    if fname == "Task":
                        logger.info(f"TASK TOOL -> gateway: {json.dumps(t, indent=2)[:2000]}")
                        break
            log_request_beautifully("POST", raw_request.url.path, display_model, gateway_body["model"], len(gateway_body["messages"]), num_tools, 200)

            if request.stream:
                # Streaming: proxy SSE from gateway → convert to Anthropic SSE
                async def gateway_stream_to_anthropic():
                    # Read timeout must be long enough for extended thinking / tool planning
                    # Session-level timeouts handle the overall limit; proxy just relays.
                    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=1800.0, write=30.0, pool=30.0)) as client:
                        async with client.stream("POST", gateway_endpoint, headers=gateway_headers, json=gateway_body) as resp:
                            if not resp.is_success:
                                err = await resp.aread()
                                raise Exception(f"Gateway error {resp.status_code}: {err.decode()[:200]}")
                            # Emit Anthropic message_start
                            yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':'msg_gw','type':'message','role':'assistant','model':original_model or request.model,'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'cache_creation_input_tokens':0,'cache_read_input_tokens':0,'output_tokens':0}}})}\n\n"
                            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
                            yield f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"

                            full_text = ""
                            tool_calls_data = []
                            content_block_idx = 0
                            current_tc_idx = -1  # Track last active tool call (don't trust index field)
                            raw_chunks = []  # Collect raw SSE for debugging
                            async for line in resp.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                if not data_str:
                                    continue
                                try:
                                    chunk = json.loads(data_str)
                                except:
                                    continue
                                raw_chunks.append(chunk)
                                choices = chunk.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})
                                finish = choices[0].get("finish_reason")

                                # Text delta
                                if delta.get("content"):
                                    text = delta["content"]
                                    full_text += text
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':text}})}\n\n"

                                # Tool calls
                                if delta.get("tool_calls"):
                                    for tc in delta["tool_calls"]:
                                        # Capture thought_signature for Gemini round-trip
                                        # Location: tc.extra_content.google.thought_signature
                                        ts = (tc.get("extra_content", {}).get("google", {}).get("thought_signature", "")
                                              or tc.get("thought_signature", ""))
                                        if ts and tc.get("id"):
                                            _cache_thought_sig(tc["id"], ts)
                                        # New tool call (has id)
                                        if tc.get("id"):
                                            # Close text block if it was open and we haven't sent tool blocks yet
                                            if not tool_calls_data:
                                                yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
                                            tool_calls_data.append({"id": tc["id"], "name": tc["function"]["name"], "arguments": ""})
                                            current_tc_idx = len(tool_calls_data) - 1
                                            # Content block index = position + 1 (index 0 is text block)
                                            block_idx = current_tc_idx + 1
                                            content_block_idx = max(content_block_idx, block_idx)
                                            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':block_idx,'content_block':{'type':'tool_use','id':tc['id'],'name':tc['function']['name'],'input':{}}})}\n\n"
                                        # Arguments delta — route to LAST active tool call
                                        # (don't trust tc.index — gateway passthrough corrupts it)
                                        if tc.get("function", {}).get("arguments"):
                                            args_delta = tc["function"]["arguments"]
                                            # Pad tool_calls_data if args arrive before id (edge case)
                                            # Route to last active tool call (current_tc_idx)
                                            # NOT tc.index which is unreliable through gateway
                                            target_idx = current_tc_idx if current_tc_idx >= 0 else 0
                                            if target_idx < len(tool_calls_data):
                                                tool_calls_data[target_idx]["arguments"] += args_delta
                                            block_idx = target_idx + 1
                                            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':block_idx,'delta':{'type':'input_json_delta','partial_json':args_delta}})}\n\n"

                                if finish:
                                    stop_reason = "tool_use" if finish == "tool_calls" else "end_turn" if finish == "stop" else finish
                                    # Fix: if response has tool_use blocks but finish_reason is "stop",
                                    # force stop_reason to "tool_use" — Gemini/GPT sometimes return "stop"
                                    # even when the response contains tool calls that need execution
                                    if stop_reason == "end_turn" and tool_calls_data:
                                        logger.warning(f"[gateway-stream] finish_reason='stop' but response has {len(tool_calls_data)} tool calls — forcing stop_reason='tool_use'")
                                        stop_reason = "tool_use"
                                    # Close last content block
                                    if not tool_calls_data:
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
                                    else:
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':content_block_idx})}\n\n"
                                    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"

                            # Safety net: if no finish_reason was seen in any chunk,
                            # emit proper closing events so CLI gets a valid SSE stream
                            if not any(choices[0].get("finish_reason") for chunk in raw_chunks for choices in [chunk.get("choices", [{}])] if choices):
                                logger.warning(f"[gateway-stream] No finish_reason in any chunk ({len(raw_chunks)} chunks, text={len(full_text)} chars, tools={len(tool_calls_data)}) — emitting default close events")
                                stop_reason = "tool_use" if tool_calls_data else "end_turn"
                                last_idx = content_block_idx if tool_calls_data else 0
                                yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':last_idx})}\n\n"
                                yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"

                            yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"
                            # Dump raw response for debugging
                            try:
                                with open("/tmp/proxy-debug-response.json", "w") as _rf:
                                    # Include raw tool_calls deltas for debugging
                                    tc_deltas = []
                                    for rc in raw_chunks:
                                        for ch in rc.get("choices", []):
                                            d = ch.get("delta", {})
                                            if d.get("tool_calls"):
                                                tc_deltas.append(d["tool_calls"])
                                    json.dump({"text": full_text[:500], "tool_calls": tool_calls_data, "num_chunks": len(raw_chunks), "has_tool_calls": bool(tool_calls_data), "raw_tc_deltas": tc_deltas[:20]}, _rf, indent=2)
                            except: pass

                return StreamingResponse(gateway_stream_to_anthropic(), media_type="text/event-stream")
            else:
                # Non-streaming: stream from gateway (Codex always streams), collect chunks
                gateway_body["stream"] = True  # Force streaming (Codex passthrough requires it)
                full_text = ""
                tool_calls_collected = []
                finish_reason = "stop"
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=1800.0, write=30.0, pool=30.0)) as client:
                    async with client.stream("POST", gateway_endpoint, headers=gateway_headers, json=gateway_body) as resp:
                        if not resp.is_success:
                            err = await resp.aread()
                            raise Exception(f"Gateway error {resp.status_code}: {err.decode()[:200]}")
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]" or not data_str:
                                continue
                            try:
                                chunk = json.loads(data_str)
                            except:
                                continue
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                full_text += delta["content"]
                            if delta.get("tool_calls"):
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    if tc.get("id"):
                                        while len(tool_calls_collected) <= idx:
                                            tool_calls_collected.append({"id": "", "name": "", "arguments": ""})
                                        tool_calls_collected[idx]["id"] = tc["id"]
                                        tool_calls_collected[idx]["name"] = tc.get("function", {}).get("name", "")
                                    if tc.get("function", {}).get("arguments"):
                                        while len(tool_calls_collected) <= idx:
                                            tool_calls_collected.append({"id": "", "name": "", "arguments": ""})
                                        tool_calls_collected[idx]["arguments"] += tc["function"]["arguments"]
                            if choices[0].get("finish_reason"):
                                finish_reason = choices[0]["finish_reason"]

                content_blocks = []
                if full_text:
                    content_blocks.append({"type": "text", "text": full_text})
                for tc in tool_calls_collected:
                    try:
                        input_data = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except:
                        input_data = {}
                    content_blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": input_data})
                stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn" if finish_reason == "stop" else finish_reason
                # Fix: if response has tool_use blocks but finish_reason is "stop",
                # force stop_reason to "tool_use" — Gemini/GPT sometimes return "stop"
                # even when the response contains tool calls that need execution
                if stop_reason == "end_turn" and tool_calls_collected:
                    logger.warning(f"[gateway-non-stream] finish_reason='stop' but response has {len(tool_calls_collected)} tool calls — forcing stop_reason='tool_use'")
                    stop_reason = "tool_use"
                return {
                    "id": "msg_gw",
                    "type": "message",
                    "role": "assistant",
                    "model": original_model or request.model,
                    "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
                }

        # For LiteLLM direct fallback (no gateway): normalize messages to OpenAI format
        # so LiteLLM's validation passes. LiteLLM converts back to provider format internally.
        if not use_gateway_direct and "messages" in litellm_request:
            litellm_request["messages"] = normalize_messages_for_gateway(litellm_request["messages"])
            logger.info(f"Normalized {len(litellm_request['messages'])} messages for LiteLLM direct")

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

            # Preserve original Claude model name for CLI compatibility
            if original_model and not original_model.startswith(('openai/', 'gemini/', 'google/')):
                request.original_model = original_model

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

            logger.debug(f"RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            # Ensure response model name matches what CLI expects (original Claude model name)
            if hasattr(anthropic_response, 'model') and original_model and not original_model.startswith(('openai/', 'gemini/', 'google/')):
                anthropic_response.model = original_model
            elif isinstance(anthropic_response, dict) and original_model and not original_model.startswith(('openai/', 'gemini/', 'google/')):
                anthropic_response['model'] = original_model
            
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

@app.post("/v1/real/{real_model_path:path}/messages/count_tokens")
@app.post("/real/{real_model_path:path}/v1/messages/count_tokens")
async def count_tokens_with_override(
    request: TokenCountRequest,
    raw_request: Request,
    real_model_path: str
):
    """Route for proxied count_tokens where real model is in URL path.
    NOTE: We remap the model to use the real model's tokenizer for more accurate counts.
    This may cause Claude CLI to prune some tools (18→12) to fit context window."""
    # Remap to real model for accurate token counting
    real_lower = real_model_path.lower()
    if 'gemini' in real_lower:
        rm = real_model_path.replace("google/", "").replace("/v1", "")
        request.model = f"gemini/{rm}"
    elif 'gpt' in real_lower or 'o1' in real_lower or 'o3' in real_lower or 'codex' in real_lower:
        rm = real_model_path.replace("/v1", "")
        request.model = f"openai/{rm}"
    return await count_tokens(request, raw_request)

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
