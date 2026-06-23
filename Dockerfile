FROM python:3.10-slim AS build

WORKDIR /app
RUN pip install --no-cache-dir uv==0.4.30

ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev


FROM python:3.10-slim

WORKDIR /app
COPY --from=build /app /app
ENV PATH=/app/.venv/bin:$PATH

EXPOSE 8082
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]
