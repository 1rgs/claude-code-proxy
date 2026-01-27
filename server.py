"""
Anthropic API Proxy for LiteLLM - Refactored with Provider Architecture

This server acts as a proxy between Anthropic API requests and various LLM backends
through LiteLLM, using a provider-based architecture for extensibility.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import os
import json
import logging
import asyncio
import sys
import uvicorn
from dotenv import load_dotenv
import litellm
from typing import Dict, Any
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Disable LiteLLM callbacks to prevent event loop blocking
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.success_callback = []
litellm.failure_callback = []

# Enable modify_params to handle thinking blocks compatibility
# When thinking_blocks are missing from assistant messages (common with OpenAI-compatible clients),
# LiteLLM will automatically drop the thinking param to avoid Anthropic 400 errors
litellm.modify_params = True

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log")]
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


# Custom filter to block LiteLLM verbose logs
class MessageFilter(logging.Filter):
    def filter(self, record):
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator",
        ]
        if hasattr(record, "msg") and isinstance(record.msg, str):
            return not any(phrase in record.msg for phrase in blocked_phrases)
        return True


root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())


# Custom colorized formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.DEBUG and "MODEL MAPPING" in record.msg:
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)


# Apply custom formatter
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(
            ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )

# Import provider architecture
from src.providers.registry import ProviderRegistry
from src.providers.base import ProviderConfig
from src.models.request import (
    MessagesRequest,
    TokenCountRequest,
    TokenCountResponse,
)
from src.utils.helpers import log_request_beautifully
from src.utils.token_count import get_robust_token_count

# Global dictionary to hold initialized providers for lifespan management
PROVIDERS: Dict[str, Any] = {}


# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Server starting up...")
    yield
    # Shutdown logic
    logger.info("Server shutting down...")
    # Close persistent clients in providers if any
    for provider_name, provider_instance in PROVIDERS.items():
        if hasattr(provider_instance, "_client") and hasattr(
            provider_instance._client, "aclose"
        ):
            logger.info(f"Closing client for provider: {provider_name}")
            await provider_instance._client.aclose()


app = FastAPI(title="Claude Code Proxy", version="1.0.0", lifespan=lifespan)

# Get environment configuration
PREFERRED_PROVIDER = os.getenv("PREFERRED_PROVIDER", "openai").lower()

# Get model configuration from environment
if PREFERRED_PROVIDER == "nvidia_nim":
    BIG_MODEL = os.getenv("BIG_MODEL", "z-ai/glm4.7")
    SMALL_MODEL = os.getenv("SMALL_MODEL", "z-ai/glm4.7")
else:
    BIG_MODEL = os.getenv("BIG_MODEL", "gpt-4.1")
    SMALL_MODEL = os.getenv("SMALL_MODEL", "gpt-4.1-mini")

# Set global configuration for model validation
MessagesRequest.set_globals(PREFERRED_PROVIDER, BIG_MODEL, SMALL_MODEL)

# Other config
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
USE_VERTEX_AUTH = os.getenv("USE_VERTEX_AUTH", "False").lower() == "true"
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "unset")

# Fast prefix detection - bypasses LLM for bash command prefix classification
# Set to "false" to disable and use the LLM for prefix detection
FAST_PREFIX_DETECTION = os.getenv("FAST_PREFIX_DETECTION", "true").lower() == "true"


def get_provider_config(provider_name: str) -> ProviderConfig:
    """Get provider configuration from environment variables."""
    if provider_name == "openai":
        return ProviderConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""), base_url=OPENAI_BASE_URL
        )
    elif provider_name == "gemini":
        return ProviderConfig(
            api_key=os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        )
    elif provider_name == "anthropic":
        return ProviderConfig(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    elif provider_name == "nvidia_nim":
        return ProviderConfig(
            api_key=os.getenv("NVIDIA_NIM_API_KEY", ""),
            base_url=os.getenv(
                "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
            ),
            rate_limit=int(os.getenv("NVIDIA_NIM_RATE_LIMIT", "40")),
        )
    elif provider_name == "nvidia_nim_direct":
        return ProviderConfig(
            api_key=os.getenv("NVIDIA_NIM_API_KEY", ""),
            base_url=os.getenv(
                "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
            ),
            rate_limit=int(os.getenv("NVIDIA_NIM_RATE_LIMIT", "40")),
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def configure_litellm_for_provider(provider_name: str):
    """Configure LiteLLM with the appropriate API key and base URL for the provider."""
    config = get_provider_config(provider_name)

    if config.api_key:
        litellm.api_key = config.api_key

    # Special handling for Vertex AI
    if provider_name == "gemini" and USE_VERTEX_AUTH:
        litellm.vertex_project = VERTEX_PROJECT
        litellm.vertex_location = VERTEX_LOCATION
        litellm.custom_llm_provider = "vertex_ai"

    # Set base URL if provided (for custom endpoints)
    if config.base_url:
        litellm.api_base = config.base_url


# Initialize the provider
try:
    provider = ProviderRegistry.get(
        PREFERRED_PROVIDER, get_provider_config(PREFERRED_PROVIDER)
    )
    PROVIDERS[PREFERRED_PROVIDER] = provider  # Store in global dictionary
    logger.info(f"Initialized provider: {PREFERRED_PROVIDER}")
except ValueError as e:
    logger.error(f"Failed to initialize provider: {e}")
    sys.exit(1)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log basic request details."""
    logger.debug(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    return response


def extract_command_prefix(command: str) -> str:
    """Extract command prefix locally for fast prefix detection.

    This avoids calling the LLM for simple prefix extraction,
    significantly speeding up permission prompts.
    """
    import shlex

    # Check for obvious command injection patterns
    injection_patterns = ['$(', '`', '&&', '||', ';', '|', '\n']
    # But || is allowed as part of fallback commands, so be more careful
    # Only flag if it looks suspicious
    if '`' in command or '$(' in command:
        return "command_injection_detected"

    try:
        # Try to parse the command
        parts = shlex.split(command)
        if not parts:
            return "none"

        # Handle environment variables at the start (e.g., FOO=bar cmd)
        env_prefix = []
        cmd_start = 0
        for i, part in enumerate(parts):
            if '=' in part and not part.startswith('-'):
                env_prefix.append(part)
                cmd_start = i + 1
            else:
                break

        if cmd_start >= len(parts):
            return "none"

        # Get the actual command parts
        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first_word = cmd_parts[0]

        # Commands that need two words for prefix (git, npm, etc.)
        two_word_commands = {'git', 'npm', 'docker', 'kubectl', 'cargo', 'go', 'pip', 'yarn'}

        if first_word in two_word_commands and len(cmd_parts) > 1:
            second_word = cmd_parts[1]
            # Don't include flags as second word
            if not second_word.startswith('-'):
                prefix = f"{first_word} {second_word}"
            else:
                prefix = first_word
        else:
            prefix = first_word

        # Add env prefix if present
        if env_prefix:
            return ' '.join(env_prefix) + ' ' + prefix

        return prefix

    except ValueError:
        # shlex parsing failed, return first word
        first_word = command.split()[0] if command.split() else "none"
        return first_word


def is_prefix_detection_request(request_data: "MessagesRequest") -> tuple[bool, str]:
    """Check if this is a Claude Code prefix detection request and extract the command.

    Claude Code sends prefix detection requests to classify bash commands for permission
    prompts. These requests have specific markers that we can detect.

    Returns (is_prefix_detection, command_string)
    """
    if len(request_data.messages) != 1:
        return False, ""

    msg = request_data.messages[0]
    if msg.role != "user":
        return False, ""

    # Get content as string
    content = ""
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        for block in msg.content:
            if hasattr(block, 'text'):
                content += block.text
            elif isinstance(block, dict) and block.get('type') == 'text':
                content += block.get('text', '')

    # Check for prefix detection markers from Claude Code
    # Pattern 1: <policy_spec>...</policy_spec> with Command: at the end
    if '<policy_spec>' in content and 'Command:' in content:
        try:
            cmd_start = content.rfind('Command:') + len('Command:')
            command = content[cmd_start:].strip()
            return True, command
        except:
            pass

    # Pattern 2: "prefix" keyword with bash command context
    # Claude Code may also send requests asking for the "prefix" of a command
    content_lower = content.lower()
    if ('prefix' in content_lower and
        ('bash' in content_lower or 'command' in content_lower) and
        len(content) < 2000):  # Prefix detection requests are typically short
        # Try to extract command from various formats
        # Look for quoted commands or commands after colons
        import re
        # Match commands in backticks, quotes, or after "Command:"
        patterns = [
            r'Command:\s*(.+?)(?:\n|$)',
            r'`([^`]+)`',
            r'"([^"]+)"',
            r"'([^']+)'",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return True, match.group(1).strip()

    return False, ""


@app.post("/v1/messages")
async def create_message(request_data: MessagesRequest, raw_request: Request):
    """Create a chat message using the configured provider."""
    try:
        # Fast path: Handle prefix detection locally without calling LLM
        # This significantly speeds up permission prompts for bash commands
        if FAST_PREFIX_DETECTION:
            is_prefix_req, command = is_prefix_detection_request(request_data)
            if is_prefix_req:
                prefix = extract_command_prefix(command)
                logger.debug(f"Fast prefix detection: '{command[:50]}...' -> '{prefix}'")

                # Return immediate response in Anthropic format
                from src.models.request import MessagesResponse, Usage
                import uuid

                response = MessagesResponse(
                    id=f"msg_{uuid.uuid4()}",
                    model=request_data.model,
                    role="assistant",
                    content=[{"type": "text", "text": prefix}],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=100, output_tokens=5),
                )
                return response

        # BEAUTIFUL LOGGING (using pydantic model directly to avoid re-parsing)
        num_messages = len(request_data.messages)
        num_tools = len(request_data.tools) if request_data.tools else 0
        original_model = request_data.model
        display_model = (
            original_model.split("/")[-1] if "/" in original_model else original_model
        )

        logger.debug(
            f"📊 PROCESSING REQUEST: Model={request_data.model}, Stream={request_data.stream}"
        )

        # Check if this is a direct provider (bypasses LiteLLM)
        is_direct_provider = hasattr(provider, "stream_response_direct")

        if is_direct_provider:
            # Direct provider - bypasses LiteLLM entirely
            # Log beautifully
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                request_data.model,
                num_messages,
                num_tools,
                200,
            )

            if request_data.stream:
                # Count input tokens for direct streaming
                # Since direct provider bypasses prepare_request, we use the build_request_body logic or robust counter
                # For NIM direct, we can use the same messages conversion or just the robust counter on original messages
                input_tokens = get_robust_token_count(
                    messages=request_data.messages,
                    model=request_data.model,
                    system=request_data.system,
                    tools=request_data.tools,
                )

                return StreamingResponse(
                    provider.stream_response_direct(
                        request_data, input_tokens=input_tokens
                    ),
                    media_type="text/event-stream",
                    headers={
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                        "Cache-Control": "no-cache",  # Prevent caching
                        "Connection": "keep-alive",
                    },
                )
            else:
                response_json = await provider.call_api_direct(request_data)
                return provider.convert_response_direct(response_json, request_data)
        else:
            # Standard LiteLLM path
            configure_litellm_for_provider(PREFERRED_PROVIDER)

            if request_data.stream:
                # All providers now have async methods
                litellm_request = await provider.prepare_streaming_request_async(
                    request_data
                )

                # Count input tokens for streaming
                # Note: system prompt is already in messages for LiteLLM path, but tools are separate
                input_tokens = get_robust_token_count(
                    messages=litellm_request["messages"],
                    model=litellm_request["model"],
                    tools=litellm_request.get("tools"),
                )

                # Include usage in stream if possible
                litellm_request["stream_options"] = {"include_usage": True}

                response_generator = await litellm.acompletion(**litellm_request)

                # Log beautifully
                num_tools = len(request_data.tools) if request_data.tools else 0
                log_request_beautifully(
                    "POST",
                    raw_request.url.path,
                    display_model,
                    litellm_request.get("model"),
                    len(litellm_request["messages"]),
                    num_tools,
                    200,
                )

                return StreamingResponse(
                    provider.stream_response(
                        response_generator,
                        input_tokens=input_tokens,
                        model=original_model,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                        "Cache-Control": "no-cache",  # Prevent caching
                        "Connection": "keep-alive",
                    },
                )
            else:
                # All providers now have async methods
                litellm_request = await provider.prepare_request_async(request_data)
                litellm_response = await litellm.acompletion(**litellm_request)
                anthropic_response = provider.convert_response(
                    litellm_response, request_data
                )

                # Log beautifully
                num_tools = len(request_data.tools) if request_data.tools else 0
                log_request_beautifully(
                    "POST",
                    raw_request.url.path,
                    display_model,
                    litellm_request.get("model"),
                    len(litellm_request["messages"]),
                    num_tools,
                    200,
                )

                return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()

        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback,
        }

        # Check for LiteLLM-specific attributes
        for attr in ["message", "status_code", "response", "llm_provider", "model"]:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        # Safe JSON serialization
        def sanitize_for_json(obj, visited=None):
            if visited is None:
                visited = set()

            obj_id = id(obj)
            if obj_id in visited:
                return f"<Circular reference to {type(obj).__name__} at {hex(obj_id)}>"

            if isinstance(obj, dict):
                visited.add(obj_id)
                return {
                    str(k): sanitize_for_json(v, visited.copy()) for k, v in obj.items()
                }
            elif isinstance(obj, (list, tuple, set)):
                visited.add(obj_id)
                return [sanitize_for_json(item, visited.copy()) for item in obj]
            elif hasattr(obj, "__dict__"):
                visited.add(obj_id)
                return sanitize_for_json(obj.__dict__, visited.copy())
            elif hasattr(obj, "text"):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        sanitized_details = sanitize_for_json(error_details)
        logger.error(
            f"Error processing request: {json.dumps(sanitized_details, indent=2)}"
        )

        error_message = f"Error: {str(e)}"
        if "message" in error_details and error_details["message"]:
            error_message += f"\nMessage: {error_details['message']}"

        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest, raw_request: Request):
    """Count tokens using the configured provider."""
    try:
        # Log the request
        original_model = request_data.original_model or request_data.model
        display_model = (
            original_model.split("/")[-1] if "/" in original_model else original_model
        )

        # Check if this is a direct provider (bypasses LiteLLM)
        is_direct_provider = hasattr(provider, "stream_response_direct")

        # Configure LiteLLM for this provider (only for non-direct providers)
        if not is_direct_provider:
            configure_litellm_for_provider(PREFERRED_PROVIDER)

        # Prepare request using provider (use async when available to avoid blocking rate limits)
        temp_request = MessagesRequest(
            model=request_data.model,
            max_tokens=100,
            messages=request_data.messages,
            system=request_data.system,
            tools=request_data.tools,
            tool_choice=request_data.tool_choice,
            thinking=request_data.thinking,
        )
        # All providers now have async methods
        litellm_request = await provider.prepare_request_async(
            temp_request, count_towards_limit=False
        )

        # Log beautifully (only for non-direct providers that return proper litellm_request)
        if not is_direct_provider:
            num_tools = len(request_data.tools) if request_data.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get("model"),
                len(litellm_request["messages"]),
                num_tools,
                200,
            )

        # Count tokens (run in thread to avoid blocking the event loop)
        # Note: system prompt is already in messages after prepare_request, but tools are separate
        token_count = get_robust_token_count(
            messages=litellm_request["messages"],
            model=litellm_request.get("model") or request_data.model,
            tools=litellm_request.get("tools"),
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
    """Root endpoint with server status."""
    return {
        "status": "ok",
        "provider": PREFERRED_PROVIDER,
        "big_model": BIG_MODEL,
        "small_model": SMALL_MODEL,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
