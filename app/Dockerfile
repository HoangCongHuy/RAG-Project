FROM python:3.11-slim
 
WORKDIR /app
 
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=10 -r requirements.txt
 
COPY . .
 
CMD ["python", "main.py"]