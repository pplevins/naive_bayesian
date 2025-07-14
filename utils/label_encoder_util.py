from sklearn.preprocessing import LabelEncoder


class LabelEncoderUtil:
    """A utility class to encode labels using a sklearn label encoder."""

    def __init__(self):
        """constructor."""
        self.encoders = {}

    def fit_transform(self, df):
        """Fit the label encoder and transform the dataset."""
        for col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        return df

    def inverse_transform(self, col_name, value):
        """Inverse the label encoding."""
        le = self.encoders.get(col_name)
        if le:
            return le.inverse_transform([value])[0]
        return value

    def get_encoder(self, col_name):
        """Get the label encoder of the given column."""
        return self.encoders.get(col_name)

    def transform_single_record(self, record_dict: dict, feature_order: list[str]) -> list[int]:
        """Transform a single record into a list of integers."""
        encoded = []
        for feature in feature_order:
            encoder = self.encoders.get(feature)
            val = record_dict[feature]
            if encoder:
                if val not in encoder.classes_ and int(val) not in encoder.classes_:
                    raise ValueError(f"Unknown value '{val}' for feature '{feature}'")
                val = encoder.transform([val])[0]
            encoded.append(val)
        return encoded
