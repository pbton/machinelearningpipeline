import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/pbton/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="pbton"
os.environ['MLFLOW_TRACKING_PASSWORD']="d25b833fe6aaa59b220a6798684e928c3a802a6f"

## Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/pbton/machinelearningpipeline.mlflow")

    ## load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)
    
    ## log metrics to MLFLOW
    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data",params["model"]])