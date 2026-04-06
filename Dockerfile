FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Force fresh start – delete any old corrupted data
RUN rm -rf /app/brain_data

RUN mkdir -p /app/brain_data

EXPOSE 7860

CMD ["uvicorn", "knowledge_graph_env:app", "--host", "0.0.0.0", "--port", "7860"]
