import pickle

from naivebayeslib import CategoricalNaiveBayes, DataLoader


class ModelState:
    """A class to store the state of the model."""

    def __init__(self):
        self.classifier = None
        self.data_loader = DataLoader()
        self.test_set = None

    def train_model(self, train_set):
        """Trains the model."""
        self.classifier = CategoricalNaiveBayes()
        self.classifier.fit(train_set.X, train_set.y)

    def store_test_set(self, test_set):
        """Stores the test set in the model state."""
        self.test_set = test_set

    def serialize_model_state(self):
        """Serializes the model state into a pickle file for sending to the prediction server."""
        return {
            "classifier": pickle.dumps(self.classifier).hex(),
            "df": pickle.dumps(self.data_loader.df).hex(),
            "encoder_util": pickle.dumps(self.data_loader.encoder_util).hex(),
            "test_set": pickle.dumps(self.test_set).hex()
        }
