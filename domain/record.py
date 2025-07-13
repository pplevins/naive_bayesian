class Record:
    """A class that represents a record in a dataset."""

    def __init__(self, features: list[str], label: str | None = None):
        """Initialize the Record class."""
        self.features = features
        self.label = label

    def __repr__(self):
        """Return a string representation of the Record."""
        return f"Record(features={self.features}, label={self.label})"
