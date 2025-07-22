import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException

from train_service.app.model_state import ModelState

app = FastAPI()
model = ModelState()


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

        accuracy = model.record_classifier.evaluate_accuracy(test_set)
        return {"message": "Model trained successfully.", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        if not model.is_model_ready():
            raise HTTPException(status_code=400, detail="Model not trained yet.")
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.file.read())
        temp.close()

        batch_dataset = model.data_loader.load_and_encode(temp.name)
        os.unlink(temp.name)

        predictions = model.record_classifier.classify_batch(batch_dataset)
        decoded_predictions = [model.data_loader.decode_prediction(pred) for pred in predictions]
        accuracy = model.record_classifier.evaluate_accuracy(batch_dataset)
        return {"predictions":
                    decoded_predictions.item() if hasattr(decoded_predictions, "item")
                    else str(decoded_predictions),
                "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/record")
async def predict_record(record: dict):
    try:
        if not model.is_model_ready():
            raise HTTPException(status_code=400, detail="Model not trained yet.")
        encoded = model.data_loader.encode_raw_input(record)
        prediction = model.record_classifier.classify_single(encoded)
        prediction_decoded = model.data_loader.decode_prediction(prediction)
        return {
            "prediction": prediction_decoded.item() if hasattr(prediction_decoded, 'item')
            else str(prediction_decoded)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features")
async def get_record_values():
    try:
        if not model.is_model_ready():
            raise HTTPException(status_code=400, detail="Model not trained yet.")
        if not model.data_loader:
            raise HTTPException(status_code=400, detail="No data is available.")

        features_dict = {}
        feature_names = model.data_loader.get_feature_names()
        for feature in feature_names:
            feature_values = model.data_loader.get_feature_unique_values(feature)
            features_dict[feature] = feature_values
            # Convert each value to a native Python type (str, int, float)
            converted_values = [v.item() if hasattr(v, 'item') else str(v) for v in feature_values]
            features_dict[feature] = converted_values
        return {"features": features_dict}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
