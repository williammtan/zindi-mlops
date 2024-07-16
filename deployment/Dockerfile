FROM --platform=linux/amd64 python:3.10-bookworm

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.2.2

# System deps:
COPY ./requirements.txt .
RUN pip install -r "requirements.txt"

# Copy only requirements to cache them in docker layer
RUN mkdir app
WORKDIR /app

COPY . /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

ENTRYPOINT ["python", "-m", "main"]