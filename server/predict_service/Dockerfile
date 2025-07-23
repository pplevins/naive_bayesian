FROM python:3.12-slim
LABEL authors="pplevins"
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY ./predict_service/app ./app
COPY ./naivebayeslib ./naivebayeslib
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8001
CMD ["uvicorn", "app.fastapi_server:app", "--host", "0.0.0.0", "--port", "8001"]