from sklearn.model_selection import train_test_split


class Dataset:
    """A class representing a dataset, and has a functionality to load and preprocess it."""

    def __init__(self, X, y):
        """Initialize the dataset."""
        self.X = X
        self.y = y

    def split(self, test_size=0.3, random_state=42):
        """Split the dataset into training and test set."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return Dataset(X_train, y_train), Dataset(X_test, y_test)
