import asyncio
import os
from pyexpat import model
from typing import Any, AsyncGenerator, Dict, List
from wsgiref import headers

from loguru import logger
from openai import OpenAI

from copilot.auth import (
    CopilotToken,
    fetch_copilot_token,
    get_base_headers,
    get_oauth_token,
)
from copilot.utils import convert_openai_to_anthropic_request


class CopilotBackend:
    copilot_token: CopilotToken = None

    def __init__(self, copilot_token: CopilotToken):
        """Initialize the CopilotBackend with an optional token."""
        self.copilot_token = copilot_token
        self.client = OpenAI(
            base_url=copilot_token.base_url_api,
            default_headers=get_base_headers(),
            api_key=copilot_token.token,
        )

    def _clean_model_name(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean the model name from kwargs.
        If the model is not provided, use the default big model.
        """
        model_name = kwargs.get("model")
        if "/" in model_name:
            kwargs["model"] = model_name.split("/")[-1]
        return kwargs

    def completion(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Function to call OpenAI API.
        """
        # logger.debug(f"Calling OpenAI API with kwargs: {kwargs}")
        clean_kwargs = self._clean_model_name(kwargs)
        return self.client.chat.completions.create(**clean_kwargs)

    async def acompletion(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """
        Async generator to call OpenAI API and yield chunks as they arrive.
        """
        # logger.debug(f"Calling OpenAI API with kwargs: {kwargs}")
        clean_kwargs = self._clean_model_name(kwargs)
        # Ensure stream=True for async generator
        clean_kwargs["stream"] = True
        response = self.client.chat.completions.create(**clean_kwargs)
        for chunk in response:
            yield chunk


if __name__ == "__main__":

    async def main():
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "What is the capital of France?",
                },
            ]
            oauth_token = await get_oauth_token()
            copilot_token = await fetch_copilot_token(oauth_token)
            copilot_backend = CopilotBackend(copilot_token=copilot_token)

            resp = copilot_backend.completion(messages=messages, model="gpt-4o-mini")
            print("Response from OpenAI API:", resp)

        except Exception as e:
            logger.exception(f"Error: {e}")

    asyncio.run(main())
