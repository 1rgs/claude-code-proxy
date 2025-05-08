FROM python:3.10-slim

RUN useradd -m appuser

WORKDIR /app
RUN chown -R appuser:appuser /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY server.py ./

RUN chown -R appuser:appuser /app

USER appuser

RUN uv venv && . .venv/bin/activate && uv pip install -e .

EXPOSE 8082

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

CMD ["/app/.venv/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]
