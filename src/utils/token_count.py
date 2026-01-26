"""Utility for robust token counting across various LLM providers."""

import logging
import json
from typing import List, Dict, Any
from litellm import token_counter
import tiktoken

logger = logging.getLogger(__name__)

# Manual mappings for models that might be missing in LiteLLM or need specific encodings
# This helps avoid some LiteLLM failures before they happen
MODEL_TO_ENCODING = {
    "nvidia/nemotron-3-nano-30b-a3b": "cl100k_base",
    "z-ai/glm4.7": "cl100k_base",  # GLM models often use similar tokenization to GPT-4
}


def get_robust_token_count(messages: List[Dict[str, Any]], model: str) -> int:
    """
    Count tokens with multiple fallback layers:
    1. litellm.token_counter (primary)
    2. tiktoken with manual mapping or cl100k_base (secondary)
    3. Character-based heuristic (final fallback)
    """
    # 1. Primary: LiteLLM
    try:
        # We wrap in a thread or just call it directly since server.py handles the threading
        return token_counter(model=model, messages=messages)
    except Exception as e:
        logger.warning(
            f"LiteLLM token_counter failed for model {model}: {str(e)}. Using fallback."
        )

    # 2. Secondary: tiktoken with manual mapping or cl100k_base
    try:
        encoding_name = MODEL_TO_ENCODING.get(model, "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)

        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        # Handle text blocks
                        text = block.get("text", "") or block.get("content", "")
                        if isinstance(text, str) and text:
                            total_tokens += len(encoding.encode(text))

                        # Handle tool results
                        if block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, str):
                                total_tokens += len(encoding.encode(tool_content))
                            elif isinstance(tool_content, list):
                                for item in tool_content:
                                    if isinstance(item, dict) and item.get("text"):
                                        total_tokens += len(
                                            encoding.encode(item["text"])
                                        )
                                    elif isinstance(item, str):
                                        total_tokens += len(encoding.encode(item))

                        # Handle tool use
                        if block.get("type") == "tool_use":
                            total_tokens += len(encoding.encode(block.get("name", "")))
                            total_tokens += len(
                                encoding.encode(json.dumps(block.get("input", {})))
                            )
                    elif isinstance(block, str):
                        total_tokens += len(encoding.encode(block))

            # Add overhead for message structure (rough estimate)
            # OpenAI typically adds ~4-5 tokens per message
            total_tokens += 4

        # Add overhead for the final response trigger
        total_tokens += 3

        logger.debug(
            f"Robust token count (tiktoken/{encoding_name}) used for model {model}: {total_tokens}"
        )
        return total_tokens
    except Exception as e:
        logger.warning(
            f"tiktoken fallback failed for model {model}: {str(e)}. Using heuristic."
        )

    # 3. Final Fallback: Heuristic (characters / 4)
    try:
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "")
                        if isinstance(text, str):
                            total_chars += len(text)

        # English often averages ~4 characters per token
        heuristic_count = max(1, total_chars // 4)
        logger.info(
            f"Heuristic token count consumed for model {model}: {heuristic_count}"
        )
        return heuristic_count
    except Exception as e:
        logger.error(f"Heuristic token count failed: {str(e)}. Returning default.")
        return 1000  # Worst case default
