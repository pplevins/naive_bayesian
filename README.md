# 🧠 Naive Bayes Classifier with CLI & FastAPI Support

A fully modular implementation of a Categorical Naive Bayes Classifier from scratch using Python.  
Supports both **interactive CLI** and **FastAPI-based REST API**, making it ideal for educational, analytical, and experimental use cases.

---

## 🚀 Features

- 📊 Categorical Naive Bayes implemented from scratch
- 📂 Load and preprocess CSV datasets with categorical or numerical features
- 🧪 70/30 train-test split with accuracy evaluation
- 🔍 Single record prediction (interactive or via API)
- 🔁 Batch prediction from CSV files
- 🔧 Modular, testable, and extensible architecture
- 🌐 FastAPI server for remote classification services
- ✅ JSON-ready responses, schema-safe

---

## 🗂️ Project Structure
```
naive_bayesian/
├── api_client.py # Client-side API interface
├── app_controller.py # CLI application controller
├── model_state.py # Singleton to hold model and encoders on the server
├── fastapi_server.py # REST API server implementation
├── core/
│ └── naive_bayes.py # Categorical Naive Bayes logic
├── loader/
│ └── data_loader.py # CSV loading and encoding utilities
├── service/
│ └── record_classifier.py # Wrapper for classification & evaluation
├── ui/
│ └── console_ui.py # User interface for CLI interaction
├── requirements.txt
└── README.md
```

---

## 🧪 CLI Usage

### ▶️ Running the App

```bash
python app.py
```

### 💡 CLI Features

- Upload and train a model on a CSV dataset
- Evaluate model on a 30% test set
- Predict a single record by entering feature values
- Predict a batch of records from another CSV file
- Supports retraining at any point
- Works with labeled categorical or integer-encoded datasets

## 🌐 API Usage
### ▶️ Start the API Server

```bash
python main_api.py
```

### 🔧 API Endpoints

| Method | Endpoint           | Description                      |
|--------|--------------------|----------------------------------|
| POST   | `/train`           | Upload CSV to train model        |
| POST   | `/predict/record`  | Predict a single record (JSON)   |
| POST   | `/predict/batch`   | Predict a batch (CSV upload)     |
| GET    | `/features`        | Get list of feature names & values |

### 📦 Example: Predict a Single Record

```python
import requests

record = {
    "age": "youth",
    "income": "high",
    "student": "no",
    "credit_rating": "fair"
}

res = requests.post("http://localhost:8000/predict/record", json=record)
print(res.json())
```

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

### 🧠 Model Internals

- Computes log prior for each class
- Computes conditional probabilities (with Laplace smoothing)
- Handles unseen categorical values at prediction time
- Works entirely from scratch (no sklearn.naive_bayes used)

### 📌 Extensibility Ideas

- Add support for numerical features via GaussianNB
- Add support for model persistence (save/load)
- Add frontend dashboard to interact with FastAPI
- Dockerize the API
