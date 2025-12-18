FROM python:3.12-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /claude-code-proxy

# Copy package specifications
COPY pyproject.toml uv.lock ./

# Install uv and project dependencies
RUN pip install --upgrade uv && uv sync --locked

# Copy project code to current directory
COPY . .

# Clean up build dependencies to reduce image size
RUN apt-get remove -y gcc \
    && apt-get autoremove -y \
    && rm -rf /root/.cargo/registry \
    && rm -rf /root/.cargo/git \
    && rm -rf /tmp/* \
    && pip cache purge

# Start the proxy
EXPOSE 8082
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]
