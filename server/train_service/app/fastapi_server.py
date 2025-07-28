import os
import tempfile

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException

from train_service.app.model_state import ModelState  # TODO: Added train_service for local running

app = FastAPI()
model = ModelState()
PREDICT_SERVICE_URL = "http://127.0.0.1:8001/load-model"


@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    """an API gateway for training a model and sending the trained model to the prediction server."""
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
