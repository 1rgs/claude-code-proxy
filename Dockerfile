FROM python:3.10-slim

WORKDIR /claude-code-proxy

# Copy package specifications
COPY pyproject.toml uv.lock ./

# Install uv and project dependencies
RUN pip install --upgrade uv && uv sync --locked

# Copy project code to current directory
COPY . .

# Start the proxy
EXPOSE 8082
CMD uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
