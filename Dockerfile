# Use a Microsoft-hosted Python image to avoid Docker Hub rate limits
FROM mcr.microsoft.com/azure-functions/python:4-python3.10

# Or fallback to Docker Hub with retry (but above is more reliable)
# FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "knowledge_graph_env:app", "--host", "0.0.0.0", "--port", "7860"]
