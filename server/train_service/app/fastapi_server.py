import os
import tempfile

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.model_state import ModelState

app = FastAPI()
model = ModelState()
PREDICT_SERVICE_URL = "http://predict_service:8001/load-model"


@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    try:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.file.read())
        temp.close()

        dataset = model.data_loader.load_and_encode(temp.name)
        os.unlink(temp.name)

        train_set, test_set = dataset.split()
        model.train_model(train_set)
        model.store_test_set(test_set)

        model_blob = model.serialize_model_state()
        response = requests.post(PREDICT_SERVICE_URL, json=model_blob)
        response.raise_for_status()
        return {"message": "Model trained and transferred successfully.", "accuracy": response.json()["accuracy"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
