import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os

from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

import mlflow

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/pbton/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="pbton"
os.environ['MLFLOW_TRACKING_PASSWORD']="d25b833fe6aaa59b220a6798684e928c3a802a6f"

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search

## Load the parameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/pbton/machinelearningpipeline.mlflow")

    ## start the MLFLOW run
    with mlflow.start_run():
        # split the dataset into training and test sets
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
        signature=infer_signature(X_train,y_train)

        ## Define the hyperparameter grid

        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [5,10,None],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2]
        }

        # Perform hyperparameter tuning
        grid_search=hyperparameter_tuning(X_train,y_train,param_grid)

        ## get the best model
        best_model=grid_search.best_estimator_

        ## predict and evaluate the model
        y_pred=best_model.predict(X_test)
        accuracy_score=accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy_score}")