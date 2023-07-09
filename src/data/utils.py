import typing as tp
from datetime import datetime, timedelta
from functools import partial

import pandas as pd
import tqdm


def is_high_season(date):
    year_date = int(date.split("-")[0])
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=year_date)
    range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=year_date)
    range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=year_date)
    range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=year_date)
    range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=year_date)
    range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=year_date)
    range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=year_date)
    range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=year_date)

    if (
        (date >= range1_min and date <= range1_max)
        or (date >= range2_min and date <= range2_max)
        or (date >= range3_min and date <= range3_max)
        or (date >= range4_min and date <= range4_max)
    ):
        return 1
    else:
        return 0


def flight_delay_minutes(flight_data):
    fecha_o = datetime.strptime(flight_data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(flight_data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    dif_min = ((fecha_o - fecha_i).total_seconds()) / 60
    return dif_min


def get_day_phase(date):
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
    morning_min = datetime.strptime("05:00", "%H:%M").time()
    morning_max = datetime.strptime("11:59", "%H:%M").time()
    evening_min = datetime.strptime("12:00", "%H:%M").time()
    evening_max = datetime.strptime("18:59", "%H:%M").time()
    night_min1 = datetime.strptime("19:00", "%H:%M").time()
    night_max1 = datetime.strptime("23:59", "%H:%M").time()
    night_min2 = datetime.strptime("00:00", "%H:%M").time()
    night_max2 = datetime.strptime("4:59", "%H:%M").time()

    if date > morning_min and date < morning_max:
        return "morning"
    elif date > evening_min and date < evening_max:
        return "evening"
    elif (date > night_min1 and date < night_max1) or (
        date > night_min2 and date < night_max2
    ):
        return "night"


def get_day_data(date) -> tp.Tuple[str, str, int]:
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    day_name = date.strftime("%A")
    month = str(date.month)
    day = date.day
    return [day_name, month, day]


def aggregate_concurrent_flights_number(
    data: pd.DataFrame, time_window: int = 1
) -> pd.DataFrame:
    string_to_date = partial(
        lambda string, _format: datetime.strptime(string, _format),
        _format="%Y-%m-%d %H:%M:%S",
    )
    data["Fecha-I"] = data["Fecha-I"].apply(string_to_date)
    data.sort_values(by="Fecha-I", inplace=True)
    dates_list: tp.List[timedelta] = data["Fecha-I"].tolist()
    concurrency_values = []
    starting_index = 0
    for i in tqdm.tqdm(range(len(dates_list))):
        concurrent_flights = 0
        for j in range(starting_index, len(dates_list)):
            time_delta = abs(dates_list[i] - dates_list[j])
            if time_delta.total_seconds() <= time_window * 3600:
                concurrent_flights += 1
            else:
                if i < j:
                    concurrency_values.append(concurrent_flights)
                    break
                else:
                    starting_index = j
            if j == len(dates_list) - 1:
                concurrency_values.append(concurrent_flights)
    data["Conc-Flights"] = concurrency_values
    return data
