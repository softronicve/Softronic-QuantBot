# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies get their own layer: they are only reinstalled when requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Unprivileged user; /app/data is the only place it can write to.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin bot \
    && mkdir /app/data \
    && chown bot:bot /app/data

COPY config/ config/
COPY src/ src/

USER bot
ENV BOT_STATE_DIR=/app/data
VOLUME ["/app/data"]

CMD ["python", "-m", "src.engine.daily_runner", "--panel"]
