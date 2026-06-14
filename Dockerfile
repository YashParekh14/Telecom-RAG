FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=error

WORKDIR /app

# Install dependencies 
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY . .

# The chroma_store/ directory must exist (run the ingest scripts, or mount it):
#   docker run -v $(pwd)/chroma_store:/app/chroma_store -p 8501:8501 --env-file .env <image>
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
