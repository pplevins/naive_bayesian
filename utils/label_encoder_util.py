from sklearn.preprocessing import LabelEncoder


class LabelEncoderUtil:
    def __init__(self):
        self.encoders = {}

    def fit_transform(self, df):
        for col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        return df

    def inverse_transform(self, col_name, value):
        le = self.encoders.get(col_name)
        if le:
            return le.inverse_transform([value])[0]
        return value

    def get_encoder(self, col_name):
        return self.encoders.get(col_name)

    def transform_single_record(self, record_dict: dict, feature_order: list[str]) -> list[int]:
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
