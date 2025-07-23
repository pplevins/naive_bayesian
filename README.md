# 🧠 Naive Bayes Classifier (Multi-container with CLI & FastAPI)

A fully modular implementation of a Categorical Naive Bayes Classifier from scratch using Python.
Supports both:

- **Command-line Interface (CLI)**
- **Microservice-based architecture via FastAPI**, with separate containers for training and prediction.

---

## 🚀 Features

- 📊 Categorical Naive Bayes implemented from scratch
- 🧪 70/30 train-test split with model evaluation
- 📥 Batch & single-record prediction
- 📂 CSV data preprocessing with category encoding/decoding
- 🔌 Internal API Gateway from `train_service` → `predict_service`
- 🐳 Dockerized & fully containerized (multi-container support)
- 🌐 RESTful FastAPI endpoints with JSON communication
- 🔐 Clean separation of logic (model, service, UI, API, etc.)
- 🔧 Modular, testable, and extensible architecture

---

## 🧱 Microservices Architecture

```
┌────────────────────┐         HTTP          ┌────────────────────┐
│   CLI / Client     │ ────────────────────▶ │   Train Service    │
└────────────────────┘                       └────────────────────┘
                                                    │
                             Internal API (Docker)  ▼
                                           ┌────────────────────┐
                                           │  Predict Service   │
                                           └────────────────────┘
```

---

## 🗂️ Project Structure

```
naive_bayes_project/
├── client/
│   ├── controller/
│   │   ├── api_client.py           # Client-side API interface
│   │   └── app_controller.py       # CLI application controller
│   ├── ui/
│   │   └── cli_interface.py        # User interface for CLI interaction
│   └── app.py
│
├── data_files/                     # CSV data files used for this project
│
├── server/
│   ├── naivebayeslib/              # Shared Python library
│   │   ├── core/
│   │   │   └── categorical_nb.py   # Categorical Naive Bayes logic
│   │   ├── loader/
│   │   │   └── data_loader.py      # CSV loading and preprocessing
│   │   ├── utils/
│   │   │   └── label_encoder_util.py  # label encoding utilities
│   │   └── __init__.py
│   │
│   ├── train_service/              # training service container
│   │   ├── app/
│   │   │   ├── fastapi_server.py   # REST API server implementation
│   │   │   └── model_state.py      # Singleton to hold model and encoders on the server
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── predict_service/
│   │   ├── app/
│   │   │   ├── fastapi_server.py   # same as in train_service
│   │   │   ├── record_classifier.py # Wrapper for classification & evaluation
│   │   │   └── model_state.py      # same as in train_service
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── compose.yaml                # docker compose file for the multi-container server
│
└── README.md
```

---

### 🌐 API Endpoints

| Service         | Method | Endpoint          | Description                        |
|-----------------|--------|-------------------|------------------------------------|
| Train Service   | POST   | `/train`          | Upload CSV and train model         |
|                 |        |                   | Sends trained model to Predict API |
| Predict Service | POST   | `/predict/record` | Predict a single record (JSON)     |
|                 | POST   | `/predict/batch`  | Predict a batch of CSV records     |
|                 | GET    | `/features`       | Return available features/values   |
|                 | POST   | `/load-model`     | Receives model blob from training  |

---

## 🐳 Docker Deployment

### ▶️ Build & Run

```bash
docker compose up --build
```

> This spins up both `train_service` (port 8000) and `predict_service` (port 8001),
> linked via internal API gateway.

---

## 🧪 CLI Usage

### ▶️ Running the App

```bash
python client/app.py
```

### 💡 CLI Features

- Upload and train a model on a CSV dataset
- Evaluate model on a 30% test set
- Predict a single record by entering feature values
- Predict a batch of records from another CSV file
- Supports retraining at any point
- Works with labeled categorical or integer-encoded datasets

### 📦 Sample API Call

```python
import requests

record = {
    "age": "youth",
    "income": "high",
    "student": "no",
    "credit_rating": "fair"
}

response = requests.post("http://localhost:8001/predict/record", json=record)
print(response.json())
```

---

## 📊 Dataset Requirements

- CSV file format
- Labels column must be the class label (named class)
- All features must be categorical or integer-based
- No missing values

### Example:

```csv
class,age,income,student,credit_rating
no,youth,high,no,fair
yes,senior,medium,yes,excellent
```

## ⚙️ Setup & Dependencies

Install dependencies

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- fastapi, uvicorn, pandas, numpy, scikit-learn

---

### 🧠 Model Internals

- Computes log prior for each class
- Computes conditional probabilities (with Laplace smoothing)
- Handles unseen categorical values at prediction time
- Works entirely from scratch (no sklearn.naive_bayes used)
- Can decode predictions back to original labels
- Modular, testable design (SRP, OOP)

### 📌 Future Extensions

- ✅ GaussianNB for numerical support
- ✅ Model persistence to disk (load/save)
- ✅ Kubernetes deployment (scalable)
- ✅ Streamlit/React frontend for web interaction
- ✅ Kafka for async prediction pipelines
