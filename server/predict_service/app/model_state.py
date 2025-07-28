import pickle

from naivebayeslib import CategoricalNaiveBayes, DataLoader
from predict_service.app.record_classifier import RecordClassifier  # TODO: Added predict_service for local running


class ModelState:
    """A class to store the state of the model."""

    def __init__(self):
        self.classifier = None
        self.data_loader = DataLoader()
        self.test_set = None
        self.record_classifier = None

    def is_model_ready(self):
        """Check if the model is trained and ready."""
        return self.classifier is not None and self.data_loader is not None

    def deserialize_model_state(self, blob: dict):
        """Deserialize the model state sent from the train server."""
        self.classifier = pickle.loads(bytes.fromhex(blob["classifier"]))
        self.data_loader.df = pickle.loads(bytes.fromhex(blob["df"]))
        self.data_loader.encoder_util = pickle.loads(bytes.fromhex(blob["encoder_util"]))
        self.test_set = pickle.loads(bytes.fromhex(blob["test_set"]))
        self.record_classifier = RecordClassifier(self.classifier)
