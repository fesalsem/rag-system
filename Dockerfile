# Intellect - Streamlit RAG document Q&A app.

# Pinned to an exact patch so CI and local builds agree on the interpreter, and
# so a new 3.12.x base cannot change behaviour underneath us between rebuilds.
FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# PyTorch first, from the CPU-only index. The default wheel drags in several GB
# of CUDA libraries this app never touches, since embeddings run on CPU.
# Installing it here means the requirements step below sees it as satisfied.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pytest is not in requirements.txt, so the runtime image has no test tooling.
# CI builds with --build-arg INSTALL_DEV=true to run the suite inside the same
# image; a plain `docker build` leaves it out.
ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ]; then \
      pip install --no-cache-dir pytest pytest-cov; \
    fi

COPY . .

RUN useradd --create-home --uid 1001 app \
    && mkdir -p /app/.cache/huggingface /tmp/faiss_index \
    && chown -R app:app /app /tmp/faiss_index
USER app

EXPOSE 8501

# Streamlit's own health endpoint. urllib raises on non-200, which is a
# non-zero exit for Docker. The start period is long because importing torch
# and sentence-transformers is slow on a cold container. Nothing is downloaded
# at startup: the MiniLM model is loaded lazily on the first query.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=4)"]

CMD ["streamlit", "run", "app.py"]
