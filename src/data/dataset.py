import typing as tp

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

from .utils import (
    aggregate_concurrent_flights_number,
    flight_delay_minutes,
    get_day_data,
    get_day_phase,
    is_high_season,
)


def load_data(data_path: str, threshold: int = 15) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file, compute derived features and add them as
    new columns. These features are computed based on functions `flight_delay_minutes`,
    `is_high_season`, `get_day_phase`, and `get_day_data`. It also applies a threshold
    on the delay_minutes to create a binary classification label (is_delayed).
    Finally, it aggregates the number of concurrent flights.

    Parameters
    ----------
    data_path : str
        The path to the CSV file that contains the dataset.
    threshold : int, optional
        The threshold (in minutes) for flight delay which determines if a flight is
        considered delayed (1) or not (0), by default 15.

    Returns
    -------
    pd.DataFrame
        The loaded data as a DataFrame with new features and labels.
    """
    data = pd.read_csv(data_path)
    data["delay_minutes"] = data.apply(flight_delay_minutes, axis=1)
    data["is_delayed"] = np.where(data["delay_minutes"] > threshold, 1, 0)
    del data["delay_minutes"]
    data["is_high_season"] = data["Fecha-I"].apply(is_high_season)
    data["day_phase"] = data["Fecha-I"].apply(get_day_phase)
    days_name, days_month, days_number = zip(*data["Fecha-I"].apply(get_day_data))
    data["Month-I"] = days_month
    data["Day-I"] = days_name
    data["Day-Number-I"] = days_number
    data = aggregate_concurrent_flights_number(data)
    return data


def build_features_and_label(
    data: pd.DataFrame, data_config: dict[str, str]
) -> tp.Tuple[pd.DataFrame, pd.Series, OneHotEncoder]:
    """
    Builds features and target from given data based on the provided configuration.
    Numerical features are selected as is, categorical features are one-hot encoded.
    The data is also balanced by downsampling the majority class in the target labels.
    The modified data, target labels, and the fitted OneHotEncoder are returned.

    Parameters
    ----------
    data : pd.DataFrame
        The original data as a DataFrame.
    data_config : dict[str, str]
        A configuration dictionary containing:
            - "features": A dictionary with "numerical" and "categorical" keys which map
              to lists of respective feature names.
            - "target": The name of the target (label) column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, OneHotEncoder]
        A tuple containing the processed data as a DataFrame, target labels as a Series,
        and the fitted OneHotEncoder.
    """
    features_dict = data_config["features"]
    numerical_feats = features_dict["numerical"]
    categorical_feats = features_dict["categorical"]
    label = data_config["target"]

    data_class_0 = data[data[label] == 0]
    data_class_1 = data[data[label] == 1]

    data_majority_downsampled = resample(
        data_class_0, replace=False, n_samples=len(data_class_1), random_state=42
    )

    data = pd.concat([data_majority_downsampled, data_class_1])
    target = data[label]
    data = data[numerical_feats + categorical_feats]

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    categorical_data_encoded = encoder.fit_transform(data[categorical_feats])

    categorical_data_encoded = pd.DataFrame(
        categorical_data_encoded,
        columns=encoder.get_feature_names_out(categorical_feats),
        index=data.index,
    )
    data = pd.concat([data[numerical_feats], categorical_data_encoded], axis=1)
    return data, target, encoder
