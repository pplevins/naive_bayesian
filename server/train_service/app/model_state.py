import pickle

from naivebayeslib import CategoricalNaiveBayes, DataLoader


class ModelState:
    def __init__(self):
        self.classifier = None
        self.data_loader = DataLoader()
        self.test_set = None

    def train_model(self, train_set):
        self.classifier = CategoricalNaiveBayes()
        self.classifier.fit(train_set.X, train_set.y)

    def store_test_set(self, test_set):
        self.test_set = test_set

    def serialize_model_state(self):
        return {
            "classifier": pickle.dumps(self.classifier).hex(),
            "df": pickle.dumps(self.data_loader.df).hex(),
            "encoder_util": pickle.dumps(self.data_loader.encoder_util).hex(),
            "test_set": pickle.dumps(self.test_set).hex()
        }
