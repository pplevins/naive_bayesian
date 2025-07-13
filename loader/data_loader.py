from utils import LabelEncoderUtil

import pandas as pd
from domain import Dataset


class DataLoader:
    """A class that loads data from csv files."""

    def __init__(self):
        """Constructor."""
        self.df = None
        self.encoder_util = None

    def get_feature_names(self) -> list[str]:
        # class column is excluded
        return [col for col in self.df.columns if col != "class"]

    def load_and_encode(self, filepath: str) -> Dataset:
        """Load and encode csv file."""
        self.df = pd.read_csv(filepath)

        # Dropping the index column
        if "Index" in self.df.columns:
            self.df.drop(columns=["Index"], inplace=True)

        # Encoding the features labels
        self.encoder_util = LabelEncoderUtil()
        df = self.encoder_util.fit_transform(self.df)

        # Splitting the dataset and the class labels column (X, y)
        X = df.drop(columns=["class"]).values
        y = df["class"].values
        return Dataset(X, y)
