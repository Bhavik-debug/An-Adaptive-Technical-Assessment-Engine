# syntax=docker/dockerfile:1

# --- builder ----------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml ./
COPY app ./app
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install .

# --- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser migrations ./migrations
# Recorded LLM responses (Day 5). Copied so that LLM_PROVIDER_ORDER=stub is a
# working mode inside the container too - a stack that runs with no provider
# credential at all is useful for a demo, and for any future CI that wants the
# whole compose stack up without spending quota.
COPY --chown=appuser:appuser fixtures ./fixtures
COPY --chown=appuser:appuser alembic.ini pyproject.toml ./

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
