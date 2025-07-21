class RecordClassifier:
    """A service class to classify a given dataset, and evaluate its accuracy."""

    def __init__(self, classifier):
        """Initialize the classifier service."""
        self.classifier = classifier

    def classify_batch(self, dataset):
        """Classify a given dataset."""
        return self.classifier.predict(dataset.X)

    def classify_single(self, encoded_record: list[int]) -> int:
        return self.classifier.predict([encoded_record])[0]

    def evaluate_accuracy(self, dataset):
        """Evaluate a given dataset."""
        return self.classifier.score(dataset.X, dataset.y)
