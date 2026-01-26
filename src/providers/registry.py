"""Provider registry for discovery and instantiation."""
from typing import Dict, Type

from .base import BaseProvider, ProviderConfig
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .anthropic import AnthropicProvider
from .nvidia_nim import NVIDIANIMProvider


class ProviderRegistry:
    """Registry for provider discovery and instantiation."""

    _providers: Dict[str, Type[BaseProvider]] = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "anthropic": AnthropicProvider,
        "nvidia_nim": NVIDIANIMProvider,
    }

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]):
        """Register a new provider."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def get(cls, name: str, config: ProviderConfig) -> BaseProvider:
        """Get a provider instance by name."""
        provider_class = cls._providers.get(name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {name}. Available providers: {list(cls._providers.keys())}")
        return provider_class(config)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered providers."""
        return list(cls._providers.keys())
