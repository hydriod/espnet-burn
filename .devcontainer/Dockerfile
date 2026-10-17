FROM mcr.microsoft.com/devcontainers/rust:1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential cmake tmux \
    && rm -rf /var/lib/apt/lists/*