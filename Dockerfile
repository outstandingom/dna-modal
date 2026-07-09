FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render dynamically assigns PORT - we must use it
EXPOSE 10000

CMD uvicorn knowledge_graph_env:app --host 0.0.0.0 --port ${PORT:-10000}
