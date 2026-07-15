FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/batmunkhcom/a2a"
LABEL org.opencontainers.image.description="A2A Protocol Runtime"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/a2a

COPY a2a/ ./a2a/
COPY plugins/ ./plugins/ 2>/dev/null || true
COPY a2a.yaml /etc/a2a/a2a.yaml 2>/dev/null || true
COPY pyproject.toml ./
COPY README.md ./

RUN pip install --no-cache-dir -e .

EXPOSE 9090 9091

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9091/health || exit 1

CMD ["a2a", "serve", "--config", "/etc/a2a/a2a.yaml"]
