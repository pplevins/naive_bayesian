import pickle

from naivebayeslib import CategoricalNaiveBayes, DataLoader
from predict_service.app.record_classifier import RecordClassifier


class ModelState:
    def __init__(self):
        self.classifier = None
        self.data_loader = DataLoader()
        self.test_set = None
        self.record_classifier = None

    def is_model_ready(self):
        return self.classifier is not None and self.data_loader is not None

    def deserialize_model_state(self, blob: dict):
        self.classifier = pickle.loads(bytes.fromhex(blob["classifier"]))
        self.data_loader.df = pickle.loads(bytes.fromhex(blob["df"]))
        self.data_loader.encoder_util = pickle.loads(bytes.fromhex(blob["encoder_util"]))
        self.test_set = pickle.loads(bytes.fromhex(blob["test_set"]))
        self.record_classifier = RecordClassifier(self.classifier)
