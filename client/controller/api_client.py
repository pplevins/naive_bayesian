import requests

BASE_URL = "http://127.0.0.1:8000"


def train_model(file_path):
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/train", files=files)
    response.raise_for_status()
    return response.json()


def classify_batch(file_path):
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/predict/batch", files=files)
    response.raise_for_status()
    return response.json()


def classify_single_record(record_dict):
    response = requests.post(f"{BASE_URL}/predict/record", json=record_dict)
    response.raise_for_status()
    return response.json()


def get_record_values():
    response = requests.get(f"{BASE_URL}/features")
    response.raise_for_status()
    return response.json()["features"]
