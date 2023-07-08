import numpy as np
import pandas as pd

from .utils import (
    flight_delay_minutes,
    get_day_of_the_month,
    get_day_phase,
    get_month_of_the_year,
    is_high_season,
)


def prepare_dataset(data_path: str, threshold: int = 15) -> pd.DataFrame:
    raw_df = pd.read_csv(data_path)
    raw_df["delay_minutes"] = raw_df.apply(flight_delay_minutes, axis=1)
    raw_df["is_delayed"] = np.where(raw_df["delay_minutes"] > threshold, 1, 0)
    del raw_df["delay_minutes"]
    raw_df["is_high_season"] = raw_df["Fecha-I"].apply(is_high_season)
    raw_df["day_phase"] = raw_df["Fecha-I"].apply(get_day_phase)
    raw_df["Day-I"] = raw_df["Fecha-I"].apply(get_day_of_the_month)
    raw_df["Month-I"] = raw_df["Fecha-I"].apply(get_month_of_the_year)
    return raw_df
