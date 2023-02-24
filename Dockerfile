FROM ubuntu:20.04

# Set lang
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Set python env vars
ENV PYTHONUNBUFFERED=1 \
    # Prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # Pip
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    \
    # Poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.1.11 \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    VENV_PATH="/opt/.venv"

ENV PATH="/opt/poetry/bin:$VENV_PATH/bin:$PATH"

# Install python 3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    git-lfs \
    ssh \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
WORKDIR /opt
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3.8 -
RUN python3.8 -m pip install --upgrade pip

# install poetry dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install

# Copy source code to the image
COPY . ./landmark-detection
ENV PYTHONPATH=/opt/landmark-detection/src:$PYTHONPATH