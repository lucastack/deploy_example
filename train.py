import json
import os
from functools import partial

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.data.dataset import build_features_and_label, load_data


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    experiment_name: str,
):

    parameters = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
    }

    model = xgb.XGBClassifier(**parameters)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    trial_f1_value = f1_score(y_true=y_val, y_pred=y_pred)

    try:
        study_best_value = trial.study.best_value
    except ValueError:
        study_best_value = 0
    if trial_f1_value > study_best_value:
        joblib.dump(
            model,
            os.path.join("models", experiment_name) + ".joblib",
        )
    return trial_f1_value


def main(n_trials: int = 100):
    data_folder = "data"
    data_csv_path = os.path.join(data_folder, "dataset_SCL.csv")
    data_config_path = os.path.join(data_folder, "config.json")

    with open(data_config_path, "r") as file:
        data_config = json.load(file)

    data = load_data(data_path=data_csv_path)
    X, y, encoder = build_features_and_label(data, data_config)

    joblib.dump(
        encoder,
        os.path.join("models", "encoder.joblib"),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    experiment_name = "xgboost_training"
    storage = f"sqlite:///optuna_db/{experiment_name}.db"

    if not os.path.exists("optuna_db"):
        os.makedirs("optuna_db")

    if not os.path.exists(f"optuna_db/{experiment_name}.db"):
        study = optuna.create_study(
            study_name=experiment_name,
            storage=storage,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True,
        )
    else:
        study = optuna.load_study(
            study_name=experiment_name,
            storage=storage,
        )

    study.optimize(
        func=partial(
            objective,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            experiment_name=experiment_name,
        ),
        n_trials=n_trials,
    )

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_model: xgb.XGBClassifier = joblib.load(
        os.path.join("models", experiment_name) + ".joblib",
    )

    y_pred = best_model.predict(X_test)
    test_score = f1_score(y_true=y_test, y_pred=y_pred)
    print(f"F1 score in test set by best model found: {test_score:.4f}")


if __name__ == "__main__":
    main(30)
