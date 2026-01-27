"""Base provider interface for all LLM backends."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator, Union
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Configuration for a provider."""

    api_key: str
    base_url: Optional[str] = None
    rate_limit: Optional[int] = None  # requests per minute
    rate_window: int = 1  # seconds

    class Config:
        arbitrary_types_allowed = True


class BaseProvider(ABC):
    """Base class for all providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def get_model_prefix(self) -> str:
        """Returns the model prefix for this provider (e.g., 'openai/', 'anthropic/')."""
        pass

    @abstractmethod
    def prepare_request(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Converts Anthropic format request to provider-specific request."""
        pass

    @abstractmethod
    def prepare_streaming_request(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Converts Anthropic format request to provider-specific streaming request."""
        pass

    async def prepare_request_async(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Async version for non-blocking request preparation.

        Default implementation calls sync version. Providers with blocking operations
        (like rate limiting) should override this with an async implementation.
        """
        return self.prepare_request(anthropic_request, count_towards_limit)

    async def prepare_streaming_request_async(
        self, anthropic_request: Any, count_towards_limit: bool = True
    ) -> Dict[str, Any]:
        """Async version for non-blocking streaming request preparation.

        Default implementation calls sync version. Providers with blocking operations
        (like rate limiting) should override this with an async implementation.
        """
        return self.prepare_streaming_request(anthropic_request, count_towards_limit)

    @abstractmethod
    def convert_response(
        self, litellm_response: Union[Dict, Any], original_request: Any
    ) -> Any:
        """Converts provider response to Anthropic format."""
        pass

    @abstractmethod
    async def stream_response(
        self,
        litellm_generator: AsyncIterator,
        input_tokens: int = 0,
        model: str = "claude-sonnet-4-20250514",
    ) -> AsyncIterator[str]:
        """Streams provider response in Anthropic SSE format."""
        pass

    def should_flatten_messages(self) -> bool:
        """Returns False by default. Override for OpenAI/NIM that need flattening."""
        return False

    def supports_native_tool_results(self) -> bool:
        """Returns whether provider supports native tool_result messages (NIM only)."""
        return False

    def get_rate_limit(self) -> tuple[int, int]:
        """Returns (rate_limit, rate_window) for this provider."""
        return (self.config.rate_limit or 0, self.config.rate_window)
