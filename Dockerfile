# The ClaimLens public showcase (M8b, ADR 0020): the web app in read-only mode, serving recorded
# sample claims. No model weights, no API key, no PyTorch: it only reads the recorded events.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

# Hugging Face Spaces runs containers as user 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --group web --group agent --no-install-project

COPY src ./src
COPY config ./config
COPY showcase ./showcase
RUN uv sync --frozen --no-dev --group web --group agent && chown -R app:app /app

USER app
# The Space's own address; any other host name is refused (DNS rebinding).
ENV CLAIMLENS_ALLOWED_HOSTS="*.hf.space,localhost,127.0.0.1" PATH="/app/.venv/bin:$PATH"
EXPOSE 7860
CMD ["claimlens", "serve", "--showcase", "--data", "showcase", "--host", "0.0.0.0", "--port", "7860"]
