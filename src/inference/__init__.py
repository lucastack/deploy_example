import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from ..data.utils import get_day_data, get_day_phase, is_high_season


class InferenceSession:
    """
    A class to represent an inference session, holding a model,
    an encoder, and data configuration. Enables preprocessing
    and prediction on new data.

    Attributes
    ----------
    model : object
        A trained model to be used for prediction.
    encoder : object
        An encoder used to transform categorical features.
    data_config : dict
        A dictionary holding the configuration of the data.

    Methods
    -------
    preprocess(data: pd.DataFrame) -> pd.DataFrame
        Preprocesses the data according to the set configuration.
    predict(data: pd.DataFrame) -> np.array
        Makes a prediction based on preprocessed data.
    """

    def __init__(self, model: XGBClassifier, encoder: OneHotEncoder, data_config: dict):
        self.model = model
        self.encoder = encoder
        self.data_config = data_config

    def preprocess(self, data: pd.DataFrame):
        """
        Preprocesses the data according to the set configuration. Applies
        necessary transformations on the input data to the model format.
        This should be a single input.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to be preprocessed.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
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
