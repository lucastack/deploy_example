import pandas as pd

from ..data.utils import get_day_data, get_day_phase, is_high_season


class InferenceSession:
    def __init__(self, model, encoder, data_config):
        self.model = model
        self.encoder = encoder
        self.data_config = data_config

    def preprocess(self, data):
        features = self.data_config["features"]
        numerical_feats = features["numerical"]
        categorical_feats = features["categorical"]
        data["day_phase"] = data["Fecha-I"].apply(get_day_phase)
        data["is_high_season"] = data["Fecha-I"].apply(is_high_season)
        days_name, days_month, days_number = zip(*data["Fecha-I"].apply(get_day_data))
        data["Month-I"] = days_month
        data["Day-I"] = days_name
        data["Day-Number-I"] = days_number
        del data["Fecha-I"]
        categorical_data_encoded = self.encoder.transform(data[categorical_feats])
        categorical_data_encoded = pd.DataFrame(
            categorical_data_encoded,
            columns=self.encoder.get_feature_names_out(categorical_feats),
            index=data.index,
        )
        data = pd.concat([data[numerical_feats], categorical_data_encoded], axis=1)
        return data

    def predict(self, data):
        preprocessed_data = self.preprocess(data)
        out = self.model.predict(preprocessed_data)
        return out
