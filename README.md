# Anthropic API Proxy for OpenAI-compatible Models

A proxy server that lets you use Anthropic clients (like Claude Code) with OpenAI, Gemini, NVIDIA NIM, or direct Anthropic backends through a modular provider architecture.

![Anthropic API Proxy](pic.png)

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for your chosen provider(s)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone https://github.com/1rgs/claude-code-proxy.git
cd claude-code-proxy
```

### Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Required variables for common setups:

| Provider | Required Variables |
|----------|-------------------|
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| NVIDIA NIM | `NVIDIA_NIM_API_KEY` |

### Running the Server

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

### Using with Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## Free Unlimited Usage with NVIDIA NIM Direct

NVIDIA NIM Direct offers **free unlimited usage** with up to **40 requests per second**. This is perfect for local development and testing.

### Getting Your Free API Key

1. Visit [https://build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
2. Sign in or create a free NVIDIA account
3. Generate a new API key (`nvapi-...`)

### Configuration

```bash
cp .env.example .env
```

Add to your `.env`:

```dotenv
PREFERRED_PROVIDER=nvidia_nim_direct
NVIDIA_NIM_API_KEY=nvapi-...
BIG_MODEL=z-ai/glm4.7
SMALL_MODEL=z-ai/glm4.7
```

The `nvidia_nim_direct` provider offers better control over streaming and thinking block handling by bypassing LiteLLM entirely, while the built-in rate limiting ensures you stay within the 40 req/s limit.

### Available Models

Popular free NVIDIA NIM models include:
- `z-ai/glm4.7` - General purpose LLM
- `moonshotai/kimi-k2-thinking` - Thinking/reasoning model
- `minimaxai/minimax-m2` - Multimodal AI
- `deepseek-ai/deepseek-v3.2` - DeepSeek V3.2
- `mistralai/devstral-2-123b-instruct-2512` - Large instruction model

---

## Configuration

### Environment Variables

<details>
<summary>Click to expand all configuration options</summary>

| Variable | Description | Default |
|----------|-------------|---------|
| `PREFERRED_PROVIDER` | Primary backend: `openai`, `gemini`, `anthropic`, `nvidia_nim`, or `nvidia_nim_direct` | `openai` |
| `BIG_MODEL` | Model mapped from `sonnet` requests | `gpt-4.1` |
| `SMALL_MODEL` | Model mapped from `haiku` requests | `gpt-4.1-mini` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GEMINI_API_KEY` | Google AI Studio API key | - |
| `USE_VERTEX_AUTH` | Use Vertex AI ADC instead of API key | `false` |
| `VERTEX_PROJECT` | Google Cloud Project ID for Vertex AI | - |
| `VERTEX_LOCATION` | Vertex AI region (e.g., `us-central1`) | - |
| `NVIDIA_NIM_BASE_URL` | NVIDIA NIM endpoint | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_NIM_API_KEY` | NVIDIA NIM API key | - |
| `NVIDIA_NIM_RATE_LIMIT` | NIM requests per second rate limit | `40` |
| `OPENAI_BASE_URL` | Custom OpenAI base URL | - |
| `FAST_PREFIX_DETECTION` | Handle bash prefix detection locally (skips LLM) | `true` |

Model mappings by provider:

| Provider | Sonnet → | Haiku → | Prefix |
|----------|----------|---------|--------|
| `openai` | `BIG_MODEL` | `SMALL_MODEL` | `openai/` |
| `gemini` | `BIG_MODEL` | `SMALL_MODEL` | `gemini/` |
| `anthropic` | `sonnet` | `haiku` | `anthropic/` |
| `nvidia_nim` | `BIG_MODEL` | `SMALL_MODEL | none |
| `nvidia_nim_direct` | `BIG_MODEL` | `SMALL_MODEL` | none |

</details>

### Example Configurations

<details>
<summary>Default OpenAI setup</summary>

```dotenv
OPENAI_API_KEY=sk-...
# PREFERRED_PROVIDER=openai # Optional, default
# BIG_MODEL=gpt-4.1
# SMALL_MODEL=gpt-4.1-mini
```

</details>

<details>
<summary>Gemini with API key</summary>

```dotenv
GEMINI_API_KEY=AIza...
PREFERRED_PROVIDER=gemini
BIG_MODEL=gemini-2.5-pro-preview-03-25
SMALL_MODEL=gemini-2.0-flash
```

</details>

<details>
<summary>Gemini with Vertex AI (ADC)</summary>

```dotenv
PREFERRED_PROVIDER=gemini
VERTEX_PROJECT=my-gcp-project
VERTEX_LOCATION=us-central1
USE_VERTEX_AUTH=true
```

</details>

<details>
<summary>Direct Anthropic proxy (just a pass-through)</summary>

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
PREFERRED_PROVIDER=anthropic
```

*Use this for logging/middleware while using real Anthropic models.*

</details>

<details>
<summary>NVIDIA NIM (via LiteLLM)</summary>

```dotenv
PREFERRED_PROVIDER=nvidia_nim
NVIDIA_NIM_API_KEY=nvapi-...
BIG_MODEL=z-ai/glm4.7
SMALL_MODEL=z-ai/glm4.7
```

</details>

<details>
<summary>NVIDIA NIM Direct (bypasses LiteLLM)</summary>

```dotenv
PREFERRED_PROVIDER=nvidia_nim_direct
NVIDIA_NIM_API_KEY=nvapi-...
BIG_MODEL=z-ai/glm4.7
SMALL_MODEL=z-ai/glm4.7
```

*Direct provider offers better control and fixes edge cases with thinking blocks and tool calling.*

</details>

## Features

### Provider Architecture

The proxy uses a modular provider system (`src/providers/`) supporting:

- **OpenAI** - Standard OpenAI API models via LiteLLM
- **Gemini** - Google AI Studio and Vertex AI via LiteLLM
- **Anthropic** - Direct Anthropic API pass-through
- **NVIDIA NIM** - Via LiteLLM (standard provider)
- **NVIDIA NIM Direct** - Direct HTTP client, bypassing LiteLLM entirely

### Thinking / Reasoning Blocks

For NVIDIA NIM direct provider, `extra_body` is passed to enable reasoning:

```json
{
  "thinking": {
    "type": "enabled",
    "clear_thinking": false
  }
}
```

The proxy streams `reasoning_content` as proper Anthropic `content_block_*` events.

### Tool Calling

All providers support tool calling with proper streaming semantics:
- Parallel tool calls
- Tool result formatting
- Streaming tool responses

### Fast Prefix Detection

When Claude Code executes bash commands, it makes a separate LLM call to classify the command prefix for permission prompts. With slow upstream providers, this can cause significant delays (30+ seconds).

The `FAST_PREFIX_DETECTION` feature (enabled by default) handles these classification requests locally without calling the LLM, providing instant permission prompts.

To disable and use the LLM for prefix detection:

```dotenv
FAST_PREFIX_DETECTION=false
```

### Log Level Control

 Server uses `WARN` level logging by default. Override:

```bash
uv run uvicorn server:app --log-level debug
```

## Docker

```bash
curl -O https://raw.githubusercontent.com/1rgs/claude-code-proxy/refs/heads/main/.env.example .env
# Edit .env with your configuration
docker run -d --env-file .env -p 8082:8082 ghcr.io/1rgs/claude-code-proxy:latest
```

Or with docker compose:

```yaml
services:
  proxy:
    image: ghcr.io/1rgs/claude-code-proxy:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - 8082:8082
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/messages` | Create a message (streaming or non-streaming) |
| `POST /v1/messages/count_tokens` | Count input tokens |
| `GET /` | Server status including provider and model config |
| `GET /health` | Health check endpoint |

## Architecture

Request flow:

1. Client sends Anthropic-format request
2. Provider translates to backend format
3. Request sent to backend (LiteLLM or direct HTTP)
4. Response converted back to Anthropic format
5. Response returned to client

For `nvidia_nim_direct`, LiteLLM is bypassed entirely for better control over streaming and thinking block handling.

## Development

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run in development mode
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
