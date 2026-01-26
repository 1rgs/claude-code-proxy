"""Request and response models for the Anthropic API."""
from pydantic import BaseModel, Field, FieldValidator as field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import logging

# Get global settings (will be populated by server.py)
PREFERRED_PROVIDER = "openai"
BIG_MODEL = "gpt-4.1"
SMALL_MODEL = "gpt-4.1-mini"
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
    "gpt-4.1",
    "gpt-4.1-mini"
]
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

logger = logging.getLogger(__name__)


# Content blocks
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


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


# Message types
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[
        Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse,
             ContentBlockToolResult, ContentBlockThinking]
    ]]
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[List[Dict[str, Any]]] = None


# Tool configuration
class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


# Request models with model mapping validator
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
    extra_body: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @classmethod
    def set_globals(cls, preferred_provider: str, big_model: str, small_model: str):
        """Set global configuration for model mapping."""
        global PREFERRED_PROVIDER, BIG_MODEL, SMALL_MODEL
        PREFERRED_PROVIDER = preferred_provider
        BIG_MODEL = big_model
        SMALL_MODEL = small_model

    @field_validator('model')
    @classmethod
    def validate_model_field(cls, v, info):
        global PREFERRED_PROVIDER, BIG_MODEL, SMALL_MODEL

        original_model = v
        new_model = v

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # Mapping logic
        mapped = False

        if PREFERRED_PROVIDER == "nvidia_nim":
            # NVIDIA NIM: no prefix; haiku/sonnet -> BIG/SMALL_MODEL, else pass-through
            if 'haiku' in clean_v.lower():
                new_model = SMALL_MODEL
                mapped = True
            elif 'sonnet' in clean_v.lower():
                new_model = BIG_MODEL
                mapped = True
            else:
                new_model = v
                mapped = True

        elif PREFERRED_PROVIDER == "anthropic":
            # Don't remap to big/small models, just add the prefix
            new_model = f"anthropic/{clean_v}"
            mapped = True

        # Map Haiku to SMALL_MODEL
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models (skip for nvidia_nim)
        elif not mapped and PREFERRED_PROVIDER != "nvidia_nim":
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True

        if mapped:
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
            if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
            new_model = v

        # Store the original model
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
    original_model: Optional[str] = None

    @field_validator('model')
    @classmethod
    def validate_model_token_count(cls, v, info):
        global PREFERRED_PROVIDER, BIG_MODEL, SMALL_MODEL

        original_model = v
        new_model = v

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # Mapping logic
        mapped = False

        if PREFERRED_PROVIDER == "nvidia_nim":
            if 'haiku' in clean_v.lower():
                new_model = SMALL_MODEL
                mapped = True
            elif 'sonnet' in clean_v.lower():
                new_model = BIG_MODEL
                mapped = True
            else:
                new_model = v
                mapped = True

        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        elif not mapped and PREFERRED_PROVIDER != "nvidia_nim":
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
            if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
            new_model = v

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
    content: List[Union[ContentBlockText, ContentBlockToolUse, ContentBlockThinking]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage
    reasoning_content: Optional[str] = None
