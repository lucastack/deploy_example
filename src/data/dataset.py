import numpy as np
import pandas as pd
from sklearn.utils import resample

from .utils import (
    aggregate_concurrent_flights_number,
    flight_delay_minutes,
    get_day_data,
    get_day_phase,
    is_high_season,
)


def load_data(data_path: str, threshold: int = 15) -> pd.DataFrame:
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
) -> pd.DataFrame:
    features_dict = data_config["features"]
    numerical_feats = features_dict["numerical"]
    categorical_feats = features_dict["categorical"]
    label = data_config["target"]

    # let's downsample class 0
    data_class_0 = data[data[label] == 0]
    data_class_1 = data[data[label] == 1]

    data_majority_downsampled = resample(
        data_class_0, replace=False, n_samples=len(data_class_1), random_state=42
    )

    data = pd.concat([data_majority_downsampled, data_class_1])
    target = data[label]

    categorical_data_encoded = pd.concat(
        [pd.get_dummies(data[feat], prefix=feat) for feat in categorical_feats], axis=1
    )
    data = pd.concat([data[numerical_feats], categorical_data_encoded], axis=1)
    return data, target
