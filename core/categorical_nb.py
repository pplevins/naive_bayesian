import numpy as np


class CategoricalNaiveBayes:
    """Naive Bayes classifier."""

    def __init__(self):
        """Initialize the classifier."""
        self.feature_log_prob = {}
        self.class_log_prior = None
        self.classes = None
        self.n_features = None
        self.n_samples = None

    def fit(self, dataset, labels):
        """Training the Naive Bayes classifier."""
        self._initialize_model(dataset, labels)
        self._compute_class_priors(labels)
        self._compute_conditional_probs(dataset, labels)

    def _initialize_model(self, dataset, labels):
        """Initialize the Naive Bayes classifier."""
        self.n_samples, self.n_features = dataset.shape
        self.classes, _ = np.unique(labels, return_counts=True)

    def _compute_class_priors(self, labels):
        """Compute the class priors."""
        _, class_counts = np.unique(labels, return_counts=True)
        self.class_log_prior = np.log(class_counts / self.n_samples)

    def _compute_conditional_probs(self, dataset, labels):
        """Compute the conditional probabilities."""
        for feature_idx in range(self.n_features):
            feature_values = np.unique(dataset[:, feature_idx])
            n_values = len(feature_values)
            for cls in self.classes:
                data_given_cls = dataset[labels == cls]
                value_counts = np.array([
                    np.sum(data_given_cls[:, feature_idx] == val)
                    for val in feature_values
                ])
                # Laplace smoothing
                probs = (value_counts + 1) / (len(data_given_cls) + n_values)
                for val, prob in zip(feature_values, probs):
                    self.feature_log_prob[(feature_idx, cls, val)] = np.log(prob)

    def predict(self, test_data):
        """Predict the class labels for the given test data."""
        return np.array([self._predict_single(sample) for sample in test_data])

    def _predict_single(self, sample):
        """Predict the class labels for the given sample."""
        log_posteriors = [
            self._compute_log_posterior(sample, class_idx, cls)
            for class_idx, cls in enumerate(self.classes)
        ]
        return self.classes[np.argmax(log_posteriors)]

    def _compute_log_posterior(self, sample, class_idx, cls):
        """Compute the log-posterior."""
        log_prob = self.class_log_prior[class_idx]
        for feature_idx, val in enumerate(sample):
            key = (feature_idx, cls, val)
            log_prob += self.feature_log_prob.get(key, np.log(1e-9))  # handle unseen values
        return log_prob

    def score(self, test_data, test_labels):
        """Calculate the accuracy of the Naive Bayes classifier."""
        return np.mean(self.predict(test_data) == test_labels)
