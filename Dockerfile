FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the app can write to brain_data directory
RUN mkdir -p /app/brain_data

EXPOSE 7860

CMD ["uvicorn", "knowledge_graph_env:app", "--host", "0.0.0.0", "--port", "7860"]
