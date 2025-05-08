# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a proxy server that allows you to use Anthropic clients (like Claude Code) with Gemini or OpenAI backends via LiteLLM. It translates API requests between Anthropic's format and OpenAI/Gemini formats, enabling Claude clients to use non-Anthropic models.

## Key Commands

### Running the Server

```bash
# Start the server with hot reloading (for development)
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload

# Start the server without hot reloading (for production)
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

### Running Tests

```bash
# Run all tests
python tests.py

# Skip streaming tests
python tests.py --no-streaming

# Run only simple tests (no tools)
python tests.py --simple

# Run only tool-related tests
python tests.py --tools-only

# Run only streaming tests
python tests.py --streaming-only
```

## Dependencies and Setup

This project uses Python 3.10+ and manages dependencies with `uv`. Key dependencies include:

- fastapi: Web framework
- uvicorn: ASGI server
- httpx: HTTP client
- pydantic: Data validation
- litellm: LLM API abstraction for model mapping
- python-dotenv: Environment variable management

## Configuration

The server uses environment variables for configuration:

1. Copy `.env.example` to `.env`
2. Configure the following variables:
   - `ANTHROPIC_API_KEY`: (Optional) Needed only if proxying *to* Anthropic models
   - `OPENAI_API_KEY`: Required if using OpenAI as provider or as fallback
   - `GEMINI_API_KEY`: Required if using Google as provider
   - `PREFERRED_PROVIDER`: Set to `openai` (default) or `google`
   - `BIG_MODEL`: The model to map `sonnet` requests to
   - `SMALL_MODEL`: The model to map `haiku` requests to

## Architecture

The proxy server works by:

1. Receiving requests in Anthropic's API format
2. Translating the requests to OpenAI/Gemini format via LiteLLM
3. Sending the translated request to the target provider
4. Converting the response back to Anthropic format
5. Returning the formatted response to the client

Key components:

- `server.py`: Main FastAPI application with all proxy logic
- `tests.py`: Test suite for verifying proxy functionality

### Model Mapping

The proxy automatically maps Claude models to either OpenAI or Gemini models:

- `haiku` → `SMALL_MODEL` with appropriate provider prefix
- `sonnet` → `BIG_MODEL` with appropriate provider prefix

The mapping logic is controlled by environment variables (see Configuration).

## Development Patterns

When working on this project, follow these guidelines:

1. Use type hints and Pydantic models for data validation
2. Log important events for debugging
3. Handle errors gracefully and provide informative error messages
4. Write tests for new functionality
5. Maintain backward compatibility with the Anthropic API
6. Keep model mappings updated as new models are released

## Code Style Guidelines

1. DO NOT add inline comments for code that is self-explanatory
2. Only add comments for complex logic that requires explanation
3. Keep code clean and readable without relying on comments
4. Use descriptive variable and function names instead of comments
5. Use comments sparingly and only when they add genuine value
6. For Docker and shell scripts, avoid obvious comments - the commands should be self-explanatory