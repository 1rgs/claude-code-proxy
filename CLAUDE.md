# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains an Anthropic API Proxy for Gemini and OpenAI Models. It enables Anthropic clients, such as Claude Code, to interface with Gemini or OpenAI models via LiteLLM, configuring the desired model backend through environment variables.

## Key Commands

### 1. Setting Up the Project
- Clone the repository:
  ```bash
  git clone https://github.com/1rgs/claude-code-openai.git
  cd claude-code-openai
  ```
- Install `uv` (if not already installed):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Configuring Environment Variables
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` to set API keys and configuration variables for models:
   - `OPENAI_API_KEY`, `GEMINI_API_KEY`, and optionally `ANTHROPIC_API_KEY`
   - `PREFERRED_PROVIDER` (default: `openai`)
   - `BIG_MODEL` and `SMALL_MODEL`

### 3. Running the Server
- Start the proxy:
  ```bash
  uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
  ```

### 4. Using with Claude Code
- Start Claude Code with the proxy:
  ```bash
  ANTHROPIC_BASE_URL=http://localhost:8082 claude
  ```

## High-Level Architecture

### 1. Project Structure
- **`src/`**: Contains main application code and modules:
  - `config.py`: Configurations for environment variables and model mapping.
  - `server.py`: Entry point for the proxy server with endpoints.
  - `copilot/`: Includes utility functions and authentication.
- **`tests.py`**: Contains test cases for core functionalities.
- **`pyproject.toml`**: Defines project dependencies and configurations.

### 2. Proxy Functionality
The proxy translates Anthropic API requests into formats compatible with OpenAI or Gemini SDKs:
1. Receives HTTP requests in Anthropic's API format.
2. Maps and prefixes model names based on configured backend (`openai/` or `gemini/`).
3. Forwards requests to the selected SDK (via LiteLLM).
4. Converts the response back into Anthropic's API format for the client.

### 3. Model Mapping
- **Default Behavior**:
  - `haiku` maps to `SMALL_MODEL`.
  - `sonnet` maps to `BIG_MODEL`.
- **Environment Variables** (in `.env`):
  - If `PREFERRED_PROVIDER=openai`, maps to OpenAI by default.
  - If `PREFERRED_PROVIDER=google`, uses Gemini models if available, with OpenAI fallback.

### Known Files
- `README.md`: Contains project setup and usage instructions.
- `.env.example`: Template for environment configuration.

### Testing
- Test suite: Run all tests to ensure functionality.
  ```bash
  pytest tests.py
  ```