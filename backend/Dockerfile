FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system --no-cache .
COPY src/ src/
CMD ["python", "-m", "cryptoswarm.main"]
