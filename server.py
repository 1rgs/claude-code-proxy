"""
Anthropic API Proxy for LiteLLM - Refactored with Provider Architecture

This server acts as a proxy between Anthropic API requests and various LLM backends
through LiteLLM, using a provider-based architecture for extensibility.
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import asyncio
import os
import logging
import json
import sys
from typing import Optional
import uvicorn
import time
import litellm

# Load environment variables
load_dotenv()

# Disable LiteLLM callbacks to prevent event loop blocking
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.success_callback = []
litellm.failure_callback = []

# Configure logging
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
            "cost_calculator"
        ]
        if hasattr(record, 'msg') and isinstance(record.msg, str):
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
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# Import provider architecture
from src.providers.registry import ProviderRegistry
from src.providers.base import ProviderConfig
from src.models.request import (
    MessagesRequest,
    TokenCountRequest,
    TokenCountResponse,
)
from src.utils.helpers import log_request_beautifully

# Initialize FastAPI app
app = FastAPI()

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


def get_provider_config(provider_name: str) -> ProviderConfig:
    """Get provider configuration from environment variables."""
    if provider_name == "openai":
        return ProviderConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=OPENAI_BASE_URL
        )
    elif provider_name == "gemini":
        return ProviderConfig(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        )
    elif provider_name == "anthropic":
        return ProviderConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY", "")
        )
    elif provider_name == "nvidia_nim":
        return ProviderConfig(
            api_key=os.getenv("NVIDIA_NIM_API_KEY", ""),
            base_url=os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            rate_limit=int(os.getenv("NVIDIA_NIM_RATE_LIMIT", "40"))
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
    provider = ProviderRegistry.get(PREFERRED_PROVIDER, get_provider_config(PREFERRED_PROVIDER))
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


@app.post("/v1/messages")
async def create_message(request_data: MessagesRequest, raw_request: Request):
    """Create a chat message using the configured provider."""
    try:
        # Log the request details
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model

        logger.debug(f"📊 PROCESSING REQUEST: Model={request_data.model}, Stream={request_data.stream}")

        # Configure LiteLLM for this provider
        configure_litellm_for_provider(PREFERRED_PROVIDER)

        # Prepare request using provider (use async when available to avoid blocking rate limits)
        if request_data.stream:
            if hasattr(provider, 'prepare_streaming_request_async'):
                litellm_request = await provider.prepare_streaming_request_async(request_data)
            else:
                litellm_request = provider.prepare_streaming_request(request_data)
            response_generator = await litellm.acompletion(**litellm_request)

            # Log beautifully
            num_tools = len(request_data.tools) if request_data.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200
            )

            return StreamingResponse(
                provider.stream_response(response_generator),
                media_type="text/event-stream"
            )
        else:
            if hasattr(provider, 'prepare_request_async'):
                litellm_request = await provider.prepare_request_async(request_data)
            else:
                litellm_request = provider.prepare_request(request_data)
            litellm_response = await litellm.acompletion(**litellm_request)
            anthropic_response = provider.convert_response(litellm_response, request_data)

            # Log beautifully
            num_tools = len(request_data.tools) if request_data.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200
            )

            return anthropic_response

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()

        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }

        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        # Safe JSON serialization
        def sanitize_for_json(obj):
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

        sanitized_details = sanitize_for_json(error_details)
        logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}")

        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"

        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest, raw_request: Request):
    """Count tokens using the configured provider."""
    try:
        from litellm import token_counter

        # Log the request
        original_model = request_data.original_model or request_data.model
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model

        # Configure LiteLLM for this provider
        configure_litellm_for_provider(PREFERRED_PROVIDER)

        # Prepare request using provider (use async when available to avoid blocking rate limits)
        temp_request = MessagesRequest(
            model=request_data.model,
            max_tokens=100,
            messages=request_data.messages,
            system=request_data.system,
            tools=request_data.tools,
            tool_choice=request_data.tool_choice,
            thinking=request_data.thinking
        )
        if hasattr(provider, 'prepare_request_async'):
            litellm_request = await provider.prepare_request_async(temp_request)
        else:
            litellm_request = provider.prepare_request(temp_request)

        # Log beautifully
        num_tools = len(request_data.tools) if request_data.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            litellm_request.get('model'),
            len(litellm_request['messages']),
            num_tools,
            200
        )

        # Count tokens (run in thread to avoid blocking the event loop)
        token_count = await asyncio.to_thread(
            token_counter,
            messages=litellm_request["messages"],
            model=litellm_request["model"]
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
        "small_model": SMALL_MODEL
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
