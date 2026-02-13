# Anthropic API Proxy for Gemini & OpenAI Models 🔄

**Use Anthropic clients (like Claude Code) with Gemini, OpenAI, or direct Anthropic backends.** 🤝

A proxy server that lets you use Anthropic clients with Gemini, OpenAI, or Anthropic models themselves (a transparent proxy of sorts), all via LiteLLM. 🌉

![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- **OpenAI API key** — for default OpenAI mapping or fallback 🔑
- **Google AI Studio (Gemini) API key** — only if using Google provider *without* Vertex auth 🔑
- **Google Cloud + Vertex AI** — if using Vertex auth (`USE_VERTEX_AUTH=true`): project with Vertex AI API enabled, and (for Claude on Vertex) Claude models enabled in [Vertex AI Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) ☁️
- **Python 3.10+** and [uv](https://github.com/astral-sh/uv) (or use `./setup_env.sh` for a venv).

### Setup 🛠️

#### From source

1. **Clone this repository**:

   ```bash
   git clone https://github.com/1rgs/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **Install uv** (if you haven't already):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure environment**:

   One universal template covers all provider modes (OpenAI, Google Gemini, Google Vertex, Anthropic):

   ```bash
   cp .env.example .env
   ```

   Edit `.env`: set API keys and choose a preset (or set variables manually). Key variables:

   - **Provider:** `PREFERRED_PROVIDER` — `openai` (default), `google`, or `anthropic`.
   - **OpenAI:** `OPENAI_API_KEY` (required for default or fallback).
   - **Google (Gemini API):** `GEMINI_API_KEY` when `PREFERRED_PROVIDER=google` and not using Vertex.
   - **Google Vertex:** `USE_VERTEX_AUTH=true`, `VERTEX_PROJECT`, `VERTEX_LOCATION`. Authenticate via **gcloud** (`gcloud auth application-default login`) and leave `VERTEX_CREDENTIALS_PATH` unset, or set `VERTEX_CREDENTIALS_PATH` to a service account JSON key. Use for Gemini or **Claude models on Vertex** (see [Vertex AI setup](#google-vertex-ai-setup) below).
   - **Models:** `BIG_MODEL` / `SMALL_MODEL` map `sonnet` / `haiku`; ignored when `PREFERRED_PROVIDER=anthropic`.
   - **Anthropic:** `ANTHROPIC_API_KEY` only when proxying directly to Anthropic.

   **Mapping:** With `openai`, models get `openai/` prefix; with `google` + Vertex auth, `vertex_ai/` (Gemini or Claude from Model Garden); with `google` and no Vertex, `gemini/` when using Gemini API key. See [Model mapping](#model-mapping️) and the presets in `.env.example`.

4. **Run the server**:

   From repo root (uv uses `.venv` by default):

   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```

   *(`--reload` is optional, for development)*  
   If you see a warning about `VIRTUAL_ENV` not matching `.venv`, you have an old virtualenv activated—run `deactivate`, then run the `uv run` command again.

   **If you used `./setup_env.sh`** (creates `.venv`): from repo root run `uv run uvicorn ...` and uv will use `.venv` with no warning:

   ```bash
   ./setup_env.sh
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```

   Or activate and run: `source .venv/bin/activate` then `uvicorn server:app --host 0.0.0.0 --port 8082`.

#### Google Vertex AI setup

When using `PREFERRED_PROVIDER=google` and `USE_VERTEX_AUTH=true`, you can use **Gemini or Claude models** on Vertex.

1. **Google Cloud:** Create or select a project and ensure billing is enabled. Enable the Vertex AI API: `gcloud services enable aiplatform.googleapis.com --project PROJECT_ID`
2. **Claude on Vertex:** In [Vertex AI Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude), open the Claude model(s) you need and click **Enable**.
3. **Authentication** — use one of these; you do **not** need both:
   - **Option A — gcloud SDK (no key file):** If your Google account has Vertex AI access on the project, log in with [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials):  
     `gcloud auth application-default login`  
     Set your project: `gcloud config set project PROJECT_ID`  
     In `.env` set only `VERTEX_PROJECT` and `VERTEX_LOCATION`; leave `VERTEX_CREDENTIALS_PATH` unset (and do not set `GOOGLE_APPLICATION_CREDENTIALS`). The proxy will use your gcloud identity.
   - **Option B — Service account JSON key:** Create a service account with at least `roles/aiplatform.user`, create a JSON key, and set `VERTEX_CREDENTIALS_PATH` in `.env` to that file path (or set `GOOGLE_APPLICATION_CREDENTIALS` externally). Use this for automation or when the machine has no interactive gcloud login.
4. Leave `GEMINI_API_KEY` unset when using Vertex.
5. **Scripts (optional):** `./setup_vertex_claude.sh -p PROJECT_ID --create-sa -y` enables the API, creates a service account and key, and writes `.env` (Option B). `./fill_env_from_gcloud.sh` fills `VERTEX_PROJECT` (and `VERTEX_CREDENTIALS_PATH` if a key file exists in the repo).

**Vertex troubleshooting:**

- **404 model not found** — Confirm the exact model ID and that the model is enabled in Model Garden for your project and region.
- **Permission denied** — With gcloud: ensure your account has Vertex AI access on the project (e.g. Vertex AI User). With a key file: ensure the service account has `roles/aiplatform.user`.
- **Location/region error** — Set `VERTEX_LOCATION` to a region supported by the model (e.g. `us-central1`).
- **Auth error** — With gcloud: run `gcloud auth application-default login` and do not set `VERTEX_CREDENTIALS_PATH`. With a key file: ensure `VERTEX_CREDENTIALS_PATH` points to a readable JSON key file.

#### Docker

If using Docker, copy the universal env template into `.env` and edit as above:

```bash
curl -o .env https://raw.githubusercontent.com/1rgs/claude-code-proxy/refs/heads/main/.env.example
```

Then, you can either start the container with [docker compose](https://docs.docker.com/compose/) (preferred):

```yml
services:
  proxy:
    image: ghcr.io/1rgs/claude-code-proxy:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - 8082:8082
```

Or with a command:

```bash
docker run -d --env-file .env -p 8082:8082 ghcr.io/1rgs/claude-code-proxy:latest
```

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):

   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:

   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use the configured backend models (defaulting to Gemini) through the proxy. 🎯

## Model Mapping 🗺️

The proxy maps Claude client aliases (`haiku` / `sonnet`) to the configured backend:

| Claude alias | Default (openai) | Google (Gemini API) | Google Vertex (`USE_VERTEX_AUTH=true`) |
| --- | --- | --- | --- |
| haiku | openai/gpt-4o-mini | gemini/[SMALL_MODEL] | vertex_ai/[SMALL_MODEL] |
| sonnet | openai/gpt-4o | gemini/[BIG_MODEL] | vertex_ai/[BIG_MODEL] |

With Vertex, `BIG_MODEL` / `SMALL_MODEL` can be **Gemini** or **Claude** model IDs from [Vertex AI Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) (e.g. `claude-sonnet-4-5@20250929`). Enable the model in Model Garden for your project and region first.

### Supported Models

#### OpenAI Models

The following OpenAI models are supported with automatic `openai/` prefix handling:

- o3-mini
- o1
- o1-mini
- o1-pro
- gpt-4.5-preview
- gpt-4o
- gpt-4o-audio-preview
- chatgpt-4o-latest
- gpt-4o-mini
- gpt-4o-mini-audio-preview
- gpt-4.1
- gpt-4.1-mini

#### Gemini Models

The following Gemini models are supported with automatic `gemini/` prefix handling (Gemini API key or Vertex):

- gemini-2.5-pro
- gemini-2.5-flash

#### Vertex AI (Gemini and Claude)

When `USE_VERTEX_AUTH=true` and `PREFERRED_PROVIDER=google`, the proxy uses the `vertex_ai/` prefix. You can set `BIG_MODEL` / `SMALL_MODEL` to:

- **Gemini** — same model IDs as above (e.g. `gemini-2.5-pro`).
- **Claude** — Model Garden IDs (e.g. `claude-sonnet-4-5@20250929`, `claude-haiku-4-5@20251001`). Enable the model in [Vertex AI Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) for your project and region.

### Model Prefix Handling

The proxy automatically adds the appropriate prefix to model names:

- OpenAI models get the `openai/` prefix
- Gemini models get the `gemini/` prefix
- Vertex models get the `vertex_ai/` prefix when `USE_VERTEX_AUTH=true` and `PREFERRED_PROVIDER=google`
- The BIG_MODEL and SMALL_MODEL prefix depends on provider/auth mode (`openai/`, `gemini/`, or `vertex_ai/`)

For example:

- `gpt-4o` becomes `openai/gpt-4o`
- `gemini-2.5-pro-preview-03-25` becomes `gemini/gemini-2.5-pro-preview-03-25`
- When BIG_MODEL is set to a Gemini model, Claude Sonnet will map to `gemini/[model-name]`
- When `USE_VERTEX_AUTH=true`, BIG_MODEL/SMALL_MODEL map to `vertex_ai/[model-name]`

### Customizing Model Mapping

Set variables in `.env` (or export them). **`.env.example`** contains one universal template with commented presets; copy it to `.env` and uncomment the block you need:

- **OpenAI (default)** — set `OPENAI_API_KEY`; optional `BIG_MODEL` / `SMALL_MODEL`.
- **Google (Gemini API)** — `PREFERRED_PROVIDER=google`, `GEMINI_API_KEY`, optional `BIG_MODEL` / `SMALL_MODEL` (e.g. `gemini-2.5-pro`, `gemini-2.5-flash`).
- **Google Vertex (Gemini)** — `PREFERRED_PROVIDER=google`, `USE_VERTEX_AUTH=true`, `VERTEX_PROJECT`, `VERTEX_LOCATION`; authenticate with gcloud (`gcloud auth application-default login`) or set `VERTEX_CREDENTIALS_PATH` to a service account key. Then set Gemini model IDs for `BIG_MODEL` / `SMALL_MODEL`.
- **Google Vertex (Claude)** — same Vertex vars and auth (gcloud or key file); set `BIG_MODEL` / `SMALL_MODEL` to Claude Model Garden IDs (e.g. `claude-sonnet-4-5@20250929`, `claude-haiku-4-5@20251001`). See [Google Vertex AI setup](#google-vertex-ai-setup).
- **Anthropic only** — `PREFERRED_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`; `BIG_MODEL` / `SMALL_MODEL` are ignored; haiku/sonnet go straight to Anthropic.

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Translating** the requests to OpenAI format via LiteLLM 🔄
3. **Sending** the translated request to OpenAI 📤
4. **Converting** the response back to Anthropic format 🔄
5. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
