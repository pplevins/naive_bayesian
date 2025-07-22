from naivebayeslib import CategoricalNaiveBayes, DataLoader
from predict_service.app.record_classifier import RecordClassifier


class ModelState:
    def __init__(self):
        self.classifier = None
        self.data_loader = DataLoader()
        self.test_set = None
        self.record_classifier = None

    def train_model(self, train_set):
        self.classifier = CategoricalNaiveBayes()
        self.classifier.fit(train_set.X, train_set.y)
        self.record_classifier = RecordClassifier(self.classifier)

    def store_test_set(self, test_set):
        self.test_set = test_set

    def is_model_ready(self):
        return self.classifier is not None and self.data_loader is not None
