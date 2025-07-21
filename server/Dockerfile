FROM python:3.12-slim
LABEL authors="pplevins"
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]