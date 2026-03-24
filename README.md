# Claude Code Proxy for MeinGPT

**Use Claude Code with any model available through MeinGPT** — GPT-5, Claude Opus, Gemini, DeepSeek, Mistral, and more.

A proxy server that translates Anthropic's Messages API into OpenAI's Chat Completions API, routing all requests through MeinGPT's OpenAI-compatible endpoint. Built on top of [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) with LiteLLM.

## How It Works

```
Claude Code ──(Anthropic API)──> Proxy :8082 ──(OpenAI API)──> MeinGPT ──> Any Model
```

1. Claude Code sends requests in Anthropic Messages API format
2. The proxy translates them to OpenAI Chat Completions format via LiteLLM
3. Requests are forwarded to MeinGPT's OpenAI-compatible API
4. Responses are translated back to Anthropic format and streamed to Claude Code

Supports streaming, tool use/function calling, and all Claude Code features.

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed
- A MeinGPT API key (get one from [app.meingpt.com](https://app.meingpt.com))

### 1. Set your MeinGPT API key as a global environment variable

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
export MEINGPT_API_KEY="sk_meingpt_YOUR_KEY_HERE"
```

Then reload: `source ~/.zshrc`

### 2. Configure the proxy

Copy the example environment file:
```bash
cp .env.example .env
```

The `.env` file is pre-configured. It loads `MEINGPT_API_KEY` from your global environment:

```env
OPENAI_API_KEY=${MEINGPT_API_KEY}
OPENAI_BASE_URL="https://app.meingpt.com/api/openai/v1"
PREFERRED_PROVIDER="openai"
BIG_MODEL="claude-opus-4-6"
SMALL_MODEL="claude-sonnet-4-5"
```

To change which models Claude Code uses, edit `BIG_MODEL` and `SMALL_MODEL` in `.env`.

> **Important:** Claude Code's system prompts and tool protocols are designed for Claude models. Non-Claude models (GPT, Gemini, etc.) may fail with skill loops, refusals, or broken tool use. Stick to Claude models for reliable operation.

### 3. Install dependencies and start the proxy

```bash
uv sync
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

### 4. Connect Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it! Claude Code now uses MeinGPT as its backend.

## Model Mapping

When Claude Code sends a request, the proxy maps models as follows:

| Claude Code Request | Maps To | Description |
|---|---|---|
| Any model with "opus" | `claude-opus-4-6` (passthrough) | Opus maps directly to itself |
| Any model with "sonnet" | `BIG_MODEL` (default: `claude-opus-4-6`) | Primary model for complex tasks |
| Any model with "haiku" | `SMALL_MODEL` (default: `claude-sonnet-4-5`) | Fast model for quick answers |

### Available MeinGPT Models

You can set `BIG_MODEL` or `SMALL_MODEL` to any of these:

**OpenAI:**
`gpt-5-4`, `gpt-5-3-chat`, `gpt-5-2`, `gpt-5-1`, `gpt-5`, `gpt-5-mini`, `gpt-4.1`, `gpt-4o`, `gpt-4o-mini`, `o3`, `o3-mini`, `o3-pro`, `o4-mini`

**Anthropic (via MeinGPT):**
`claude-opus-4-6`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-sonnet-4`, `claude-opus-4-5`

**Google:**
`gemini-3.1-pro`, `gemini-3-flash`, `gemini-2.5-pro`, `gemini-2.5-flash`

**Other:**
`deepseek-r1`, `deepseek-v3`, `llama-3.3-fast`, `mistral-medium`, `codestral`

## Configuration

All configuration is done via the `.env` file:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | MeinGPT API key (loaded from `$MEINGPT_API_KEY`) | — |
| `OPENAI_BASE_URL` | MeinGPT API endpoint | `https://app.meingpt.com/api/openai/v1` |
| `PREFERRED_PROVIDER` | Provider routing for LiteLLM | `openai` |
| `BIG_MODEL` | Model for sonnet/complex requests | `claude-opus-4-6` |
| `SMALL_MODEL` | Model for haiku/fast requests | `claude-sonnet-4-5` |

## Troubleshooting

### "Cannot POST .../chat/completions/chat/completions"
`OPENAI_BASE_URL` must end at `/v1` — LiteLLM automatically appends `/chat/completions`.

### "OPENAI_API_KEY is empty"
Ensure `MEINGPT_API_KEY` is exported in your shell environment before starting the proxy.

### Proxy not responding
Check if it's running: `curl http://localhost:8082/`

Kill and restart:
```bash
lsof -ti :8082 | xargs kill -9
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

## Architecture

This proxy is built on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by 1rgs, adapted for MeinGPT. Key components:

- **FastAPI** — HTTP server handling Anthropic API requests
- **LiteLLM** — Translates between Anthropic and OpenAI API formats
- **Pydantic** — Request/response validation
- **Streaming** — Full SSE support for real-time token streaming

The proxy handles content block conversion (Anthropic uses `[{type: "text", text: "..."}]`, OpenAI uses plain strings), tool use translation (Anthropic tool_use/tool_result ↔ OpenAI function calling), and model name mapping.

## Credits

Based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by [1rgs](https://github.com/1rgs).
