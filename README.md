
# Flight Delay Prediction Project
## Description

This project aims to predict flight delays using machine learning models and flight data. By training the model on historical flight data, we can accurately predict whether a flight will be delayed.
## Features

- Data loading and preprocessing: Load data from CSV files, and perform feature engineering. Details can be seen in `load_data` function in `src/data/dataset`. 

- Model training and evaluation: Train a XGBoost model and evaluate its performance using F1-score.
Uses Optuna to find best hyperparamenters (based on validation dataset).

- Deployment in Fast API. Endpoint `predict/`. 

## How to Use

### Training

Note: you need to have .csv file inside /data folder and requirements installed.

    python train.py


![alt text](figures/train.png)

### Deploy

Note: you need to have a trained model in models/. That means, to have `encoder.joblib` and `xgboost_training.joblib`.

    python api.py

### Deploy + docker

    docker build -t deploy_container .
    docker run -p 8000:8080 deploy_container

### Test 

Using wrk we can test the endpoint as you can see

![alt text](figures/test_endpoint.png) 

which results in 80.000 requests in 45s

�