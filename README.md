
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


We achieve 67% in test set with XGBoost using downsampling to produce a balanced dataset.

### Deploy (locally)
 
Note: you need to have a trained model in models/. That means, to have `encoder.joblib` and `xgboost_training.joblib`.

    python api.py

### Deploy + docker (locally)

    docker build -t deploy_container .
    docker run -p 8000:8080 deploy_container

### Deploy in GCP

Deployment is done using Cloud Build, which is done in two steps: first, build docker image and push it to Artifacts registry and two, deploy the app in a container using Cloud Run.

The resource (essentially a trigger) can be created using the Terraform config provided in this repo (see `cloudbuild_trigger.tf`). You need to set location variable as your GCP project location.


### Test Endpoint (GCP)

Using wrk we can test the endpoint as you can see

![alt text](figures/test_endpoint.png) 

which results in 80.000 requests in 45s
�