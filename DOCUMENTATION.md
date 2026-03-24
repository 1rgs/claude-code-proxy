# Claude Code Proxy for MeinGPT — Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Request Flow](#request-flow)
4. [API Format Translation](#api-format-translation)
5. [Model Mapping](#model-mapping)
6. [Streaming](#streaming)
7. [Tool Use / Function Calling](#tool-use--function-calling)
8. [Configuration Reference](#configuration-reference)
9. [MeinGPT API Details](#meingpt-api-details)
10. [Development Guide](#development-guide)
11. [Related Projects](#related-projects)

---

## Overview

This proxy enables Claude Code to use any model available through MeinGPT by translating between two API formats:

- **Input:** Anthropic Messages API (what Claude Code speaks)
- **Output:** OpenAI Chat Completions API (what MeinGPT exposes)

The proxy is a modified version of [claude-code-proxy](https://github.com/1rgs/claude-code-proxy), adapted to route through MeinGPT instead of direct provider APIs.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Claude Code CLI                           │
│  Sends Anthropic Messages API requests to ANTHROPIC_BASE_URL     │
└──────────────────────┬───────────────────────────────────────────┘
                       │ POST /v1/messages
                       │ (Anthropic format)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Proxy Server (:8082)                           │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ FastAPI Router   │→ │ Pydantic     │→ │ Model Validator    │  │
│  │ /v1/messages     │  │ Validation   │  │ sonnet→BIG_MODEL   │  │
│  │ /v1/messages/    │  │              │  │ haiku→SMALL_MODEL  │  │
│  │  count_tokens    │  │              │  │ + openai/ prefix   │  │
│  └─────────────────┘  └──────────────┘  └────────────────────┘  │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ convert_anthropic_to_litellm()                              │ │
│  │ - System message extraction                                 │ │
│  │ - Content block → plain string conversion                   │ │
│  │ - Tool format translation (Anthropic → OpenAI functions)    │ │
│  │ - max_tokens capping (16384 for OpenAI models)              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ LiteLLM acompletion() / completion()                        │ │
│  │ - Strips openai/ prefix                                     │ │
│  │ - Sends to api_base (MeinGPT)                               │ │
│  │ - Handles streaming chunks                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ convert_litellm_to_anthropic() / handle_streaming()         │ │
│  │ - OpenAI response → Anthropic Messages response             │ │
│  │ - SSE event generation for streaming                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────────────┘
                       │ OpenAI Chat Completions format
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     MeinGPT API                                  │
│  https://app.meingpt.com/api/openai/v1/chat/completions          │
│  Bearer token auth with sk_meingpt_... key                       │
│  Routes to: GPT-5, Claude Opus, Gemini, DeepSeek, etc.           │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| File | Purpose |
|---|---|
| `server.py` | Main proxy server — FastAPI app, request handling, format translation |
| `.env` | Configuration — API key, endpoint URL, model mapping |
| `pyproject.toml` | Dependencies — FastAPI, LiteLLM, httpx, uvicorn |

---

## Request Flow

### Non-Streaming Request

1. Claude Code sends `POST /v1/messages` with Anthropic-format body
2. Pydantic `MessagesRequest` validates and maps the model name
3. `convert_anthropic_to_litellm()` transforms the request
4. `litellm.completion()` sends to MeinGPT
5. `convert_litellm_to_anthropic()` transforms the response
6. Returns JSON in Anthropic Messages format

### Streaming Request

1. Same as above through step 3
2. `litellm.acompletion()` returns an async generator
3. `handle_streaming()` converts each OpenAI chunk to Anthropic SSE events
4. Returns `StreamingResponse` with `text/event-stream` media type

---

## API Format Translation

### Content Blocks

Anthropic uses structured content blocks; OpenAI uses plain strings:

```
Anthropic:  {"content": [{"type": "text", "text": "Hello"}]}
OpenAI:     {"content": "Hello"}
```

The proxy handles this conversion in both directions.

### System Messages

Anthropic sends system messages as a separate `system` field; OpenAI puts them in the messages array:

```
Anthropic:  {"system": "You are helpful", "messages": [...]}
OpenAI:     {"messages": [{"role": "system", "content": "You are helpful"}, ...]}
```

### Usage/Token Counts

```
Anthropic:  {"usage": {"input_tokens": 100, "output_tokens": 50}}
OpenAI:     {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
```

---

## Model Mapping

### How It Works

The Pydantic `field_validator` on the `model` field (`server.py:200-264`) handles mapping:

1. Claude Code sends model names like `claude-sonnet-4-6` or `claude-haiku-4-5-20251001`
2. The validator detects keywords:
   - `"opus"` in name → maps to `openai/claude-opus-4-6` (passthrough)
   - `"sonnet"` in name → maps to `openai/{BIG_MODEL}`
   - `"haiku"` in name → maps to `openai/{SMALL_MODEL}`
3. LiteLLM uses the `openai/` prefix to route to the OpenAI provider
4. LiteLLM strips the `openai/` prefix before sending to MeinGPT
5. MeinGPT receives the bare model name (e.g., `claude-opus-4-6`)

### Known Model Lists

Models must be in `OPENAI_MODELS` or `GEMINI_MODELS` lists (`server.py:104-137`) for the prefix logic to work correctly. MeinGPT models have been added to these lists.

### Current Default Mapping

| Claude Code Sends | Proxy Maps To | MeinGPT Receives |
|---|---|---|
| `claude-opus-4-6` | `openai/claude-opus-4-6` | `claude-opus-4-6` (passthrough) |
| `claude-sonnet-4-6` | `openai/claude-opus-4-6` | `claude-opus-4-6` (via BIG_MODEL) |
| `claude-haiku-4-5-20251001` | `openai/claude-sonnet-4-5` | `claude-sonnet-4-5` (via SMALL_MODEL) |

---

## Streaming

### SSE Event Format

The proxy translates OpenAI streaming chunks to Anthropic SSE events:

```
event: message_start
data: {"type": "message_start", "message": {...}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10}}

event: message_stop
data: {"type": "message_stop"}
```

### Streaming Implementation

`handle_streaming()` (`server.py:833-1093`):
- Accumulates text content across chunks
- Handles text-to-tool transitions (closes text block when tool calls appear)
- Sends `input_json_delta` events for tool call arguments
- Maps OpenAI `finish_reason` to Anthropic `stop_reason`

---

## Tool Use / Function Calling

### Translation

Anthropic tool format ↔ OpenAI function calling:

**Anthropic (input):**
```json
{
  "tools": [{
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {"type": "object", "properties": {...}}
  }]
}
```

**OpenAI (sent to MeinGPT):**
```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather",
      "parameters": {"type": "object", "properties": {...}}
    }
  }]
}
```

### Tool Results

Tool results in user messages (Anthropic `tool_result` blocks) are extracted and converted to plain text for OpenAI compatibility.

---

## Configuration Reference

### Environment Variables (`.env`)

| Variable | Required | Description | Default |
|---|---|---|---|
| `MEINGPT_API_KEY` | Yes (global) | MeinGPT API key, must be set in shell env | — |
| `OPENAI_API_KEY` | Auto | Set to `${MEINGPT_API_KEY}` in `.env` | — |
| `OPENAI_BASE_URL` | Yes | MeinGPT endpoint | `https://app.meingpt.com/api/openai/v1` |
| `PREFERRED_PROVIDER` | No | LiteLLM provider routing | `openai` |
| `BIG_MODEL` | No | Model for sonnet/complex requests | `claude-opus-4-6` |
| `SMALL_MODEL` | No | Model for haiku/fast requests | `claude-sonnet-4-5` |

### Important Notes

- **`OPENAI_BASE_URL` must end at `/v1`** — LiteLLM appends `/chat/completions` automatically
- **`MEINGPT_API_KEY` must be a global env var** — The `.env` references it via `${MEINGPT_API_KEY}`
- The proxy runs on port **8082** by default

---

## MeinGPT API Details

### Endpoint
```
POST https://app.meingpt.com/api/openai/v1/chat/completions
```

### Authentication
```
Authorization: Bearer sk_meingpt_...
```

### Request Format
Standard OpenAI Chat Completions API:
```json
{
  "model": "claude-opus-4-6",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "max_tokens": 4096,
  "stream": true
}
```

### Response Format
Standard OpenAI Chat Completions response with streaming support (SSE chunks).

### Supported Models (as of March 2026)

**OpenAI:** gpt-5-4, gpt-5-3-chat, gpt-5-2, gpt-5-1, gpt-5, gpt-5-mini, gpt-4.1, gpt-4o, o3, o3-pro, o4-mini

**Anthropic:** claude-opus-4-6, claude-sonnet-4-6, claude-sonnet-4-5, claude-sonnet-4, claude-opus-4-5

**Google:** gemini-3.1-pro, gemini-3-flash, gemini-2.5-pro, gemini-2.5-flash

**Other:** deepseek-r1, deepseek-v3, llama-3.3-fast, mistral-medium, codestral, sonar, minimax-m2.1, kimi-k2-instruct

---

## Development Guide

### Running Locally

```bash
# Set your MeinGPT API key
export MEINGPT_API_KEY="sk_meingpt_YOUR_KEY"

# Install dependencies
uv sync

# Start the proxy
uv run uvicorn server:app --host 0.0.0.0 --port 8082

# In another terminal, connect Claude Code
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

### Testing with curl

**Non-streaming:**
```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

**Streaming:**
```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 100,
    "stream": true,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

**Health check:**
```bash
curl http://localhost:8082/
```

### Adding New Models

To support a new MeinGPT model, add it to the appropriate list in `server.py`:

- OpenAI/Anthropic/other models → `OPENAI_MODELS` list (~line 104)
- Gemini models → `GEMINI_MODELS` list (~line 131)

This ensures the model validator adds the correct `openai/` or `gemini/` prefix for LiteLLM routing.

### Changing Default Models

Edit `.env`:
```env
BIG_MODEL="claude-opus-4-6"        # For sonnet requests
SMALL_MODEL="gemini-2.5-flash"  # For haiku requests
```

Restart the proxy for changes to take effect.

---

## Related Projects

| Project | Location | Purpose |
|---|---|---|
| **meingpt-api** | `../meingpt-api/` | MCP server + simple proxy for MeinGPT |
| **meingpt_mcp** | `../meingpt_mcp/` | Full MCP server with browser automation + tRPC HTTP client |
| **claude-code-proxy** (upstream) | [github.com/1rgs/claude-code-proxy](https://github.com/1rgs/claude-code-proxy) | Original proxy this is based on |

### Differences from upstream claude-code-proxy

| Feature | Upstream | This Version |
|---|---|---|
| Backend | Direct OpenAI/Gemini/Anthropic APIs | MeinGPT (unified gateway) |
| API Key | Separate per-provider keys | Single `MEINGPT_API_KEY` |
| Model List | Standard OpenAI + Gemini models | Extended with MeinGPT models |
| Config | Multiple provider options | Simplified for MeinGPT |
