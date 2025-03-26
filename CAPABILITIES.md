# Claude to Alternative Models Capability Mapping

This document provides a comprehensive reference for how Claude models and capabilities map to their Deepseek and Google Gemini equivalents when using the proxy, based on recent testing and performance analysis.

## Model Mapping

| Claude Capability | Alternative Equivalent | Provider | Notes |
|-------------------|------------------------|----------|-------|
| **Core Models** | | | |
| Claude Haiku | gemini-2.0-flash | Google | Default for simpler tasks and quick responses |
| Claude Sonnet | deepseek-chat | Deepseek | Default for complex tasks and code generation |
| Claude Opus | Not directly mapped | - | No direct equivalent in current lineup |
| **Specialized Tasks** | | | |
| Code generation | deepseek-chat | Deepseek | High-quality code generation in multiple languages |
| Reasoning | deepseek-chat + CoT prompt | Deepseek | Excellent with Chain-of-Thought system prompt enabled |
| Quick answers | gemini-2.0-flash | Google | Very fast responses (~0.9s) for simple queries |
| Tool usage | gemini-2.0-flash | Google | Fast tool/function calling (~0.6s) |
| **Technical Capabilities** | | | |
| Function calling/Tool use | Fully supported by both | Both | Complete support for OpenAI-compatible function calling |
| Streaming responses | Fully supported by both | Both | All required event types present in SSE format |
| System prompts | Fully supported by both | Both | Properly processed as first message in conversation |
| **Context Windows** | | | |
| Deepseek | 8192 tokens | Deepseek | Smaller than Claude Sonnet (200K) and Opus (1M) |
| Gemini | 1,048,576 input / 8,192 output | Google | Large input context but limited output tokens |
| **Integration** | | | |
| API format | OpenAI-compatible | Both | Similar format to OpenAI, works with LiteLLM |
| Base URLs | https://api.deepseek.com | Deepseek | Default API endpoints |
| | https://generativelanguage.googleapis.com | Google | |
| API parameters | OpenAI-compatible | Both | Temperature, top_p, top_k, similar to OpenAI/Claude |
| **Content Types** | | | |
| Text generation | Fully supported by both | Both | High-quality text responses |
| Code generation | Fully supported by both | Both | Strong performance in Deepseek models (~20s) |
| Complex reasoning | Fully supported | Deepseek | Excellent with CoT but slower (~43s) |
| Multi-turn conversations | Fully supported by both | Both | Maintains context across messages |
| **Developer Tools** | | | |
| Integration | LiteLLM | Both | Common integration framework |
| Utilities | Function calling frameworks | Both | Support for structured tool use |

## Test Results

Recent comprehensive testing verifies compatibility with all core Claude Code capabilities:

| Test Case | Status | Performance | Notes |
|-----------|--------|-------------|-------|
| Simple text generation | ✅ PASS | 0.7-0.8s | Gemini provides fast, high-quality responses |
| Calculator tool usage | ✅ PASS | 0.6-0.7s | Proper function calling format |
| Multiple tool usage | ✅ PASS | 0.6-0.7s | Successfully handles multiple tools in sequence |
| Multi-turn conversation | ✅ PASS | 0.5-0.6s | Maintains context across messages |
| Complex content blocks | ✅ PASS | 0.5-0.6s | Correctly processes different content types |
| Chain-of-Thought reasoning | ✅ PASS | 16-17s | Successfully solves complex problems with step-by-step reasoning |
| Code generation | ✅ PASS | 18-19s | Generates correct, well-formatted code |
| Streaming text | ✅ PASS | Real-time | All required event types present |
| Streaming with tools | ✅ PASS | Real-time | Proper handling of streaming tool calls |

## Key Considerations

When using alternative models with Claude Code through this proxy:

1. **Model Selection**: 
   - **Haiku models** map to `gemini-2.0-flash` (general tasks, quick responses, tool usage)
   - **Sonnet models** map to `deepseek-chat` (coding tasks, complex reasoning)
   - The Gemini Flash model is surprisingly fast and effective for simpler tasks
   - Deepseek is excellent for code generation and reasoning, but significantly slower

2. **Recommended Settings**: 
   - Always use the `--always-cot` flag when starting the server
   - This ensures Chain-of-Thought prompting for all Sonnet/Deepseek requests
   - Significantly improves reasoning capability for complex tasks

3. **Token Limits**: 
   - Both models have output token limits of 8192 tokens (enforced automatically)
   - Gemini models have a large input context window (over 1M tokens)

4. **Function Calling**: 
   - Both models fully support function calling in OpenAI-compatible format
   - Gemini Flash is particularly fast for tool usage (~0.6s response time)

5. **Performance Tradeoffs**: 
   - Gemini Flash: Extremely fast (0.5-0.8s) for simple tasks, chat, and tool usage
   - Deepseek: Slower but high-quality for reasoning (16-17s) and code generation (18-19s)
   - These times are from latest benchmark tests run in March 2025
   - Consider your use case when choosing which model to use

6. **Customization**:
   - Override default mapping through environment variables in `.env` file
   - The `--always-cot` flag is highly recommended for improving reasoning