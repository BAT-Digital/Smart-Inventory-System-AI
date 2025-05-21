 
FROM python:3.10-slim

 
RUN apt-get update && apt-get install -y \
    build-essential \
    libpython3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

 
WORKDIR /app


COPY requirements.txt .
COPY app.py .
COPY forecast_model.py .

 
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

 
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
