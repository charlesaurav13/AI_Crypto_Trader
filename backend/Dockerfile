FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN uv pip install --system --no-cache .
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "cryptoswarm.main"]
