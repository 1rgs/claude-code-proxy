# Claude Code Proxy for NVIDIA NIM

Use **Claude Code CLI for free** with NVIDIA NIM's unlimited API. This lightweight proxy converts Claude Code's Anthropic API requests to NVIDIA NIM format.

## Quick Start

### 1. Get Your Free NVIDIA API Key

1. Visit [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
2. Sign in or create a free NVIDIA account
3. Generate a new API key (`nvapi-...`)

### 2. Install & Configure

```bash
git clone https://github.com/Alishahryar1/claude-code-proxy.git
cd claude-code-proxy

cp .env.example .env
```

Edit `.env`:

```dotenv
NVIDIA_NIM_API_KEY=nvapi-your-key-here
BIG_MODEL=qwen/qwen2.5-coder-32b-instruct
SMALL_MODEL=qwen/qwen2.5-coder-32b-instruct
```

### 3. Run

**Terminal 1 - Start the proxy:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2 - Run Claude Code:**

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it! Claude Code now uses NVIDIA NIM for free.

## Available Models

| Model | Description |
|-------|-------------|
| `qwen/qwen2.5-coder-32b-instruct` | Best for coding tasks |
| `deepseek-ai/deepseek-r1` | DeepSeek R1 reasoning |
| `nvidia/llama-3.3-nemotron-super-49b-v1` | NVIDIA's Llama variant |
| `meta/llama-3.1-405b-instruct` | Meta's largest Llama |

Browse all models at [build.nvidia.com](https://build.nvidia.com/explore/discover)

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_NIM_API_KEY` | Your NVIDIA API key | required |
| `BIG_MODEL` | Model for Sonnet/Opus requests | `qwen/qwen2.5-coder-32b-instruct` |
| `SMALL_MODEL` | Model for Haiku requests | `qwen/qwen2.5-coder-32b-instruct` |
| `NVIDIA_NIM_BASE_URL` | NIM endpoint | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_NIM_RATE_LIMIT` | Requests per minute | `40` |
| `FAST_PREFIX_DETECTION` | Skip LLM for bash prefix detection | `true` |

## Adding Your Own Provider

Extend `BaseProvider` in `providers/` to add support for other APIs:

```python
from providers.base import BaseProvider, ProviderConfig

class MyProvider(BaseProvider):
    async def complete(self, request):
        # Make API call, return raw JSON
        pass

    async def stream_response(self, request, input_tokens=0):
        # Yield Anthropic SSE format events
        pass

    def convert_response(self, response_json, original_request):
        # Convert to Anthropic response format
        pass
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/messages` | Chat completion (streaming/non-streaming) |
| `POST /v1/messages/count_tokens` | Token counting |
| `GET /` | Server status |
| `GET /health` | Health check |
