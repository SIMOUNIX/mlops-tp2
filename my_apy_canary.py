from fastapi import FastAPI
from pydantic import BaseModel
import random

from model_utils import make_prediction, load_model

model_v1 = load_model("tracking-quickstart", "1")
model_v2 = load_model("tracking-quickstart", "2")

# Iris dataset 
class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
app = FastAPI()

@app.post("/predict")
def predict(features: Features):
    print(f"Received features: {features}")
    model = model_v1
    
    # canary deployment 10% model_v2
    if random.random() < 0.1:
        model = model_v2
    else:
        model = model_v1
    
    # # it has to be a 2d array
    input_data = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]

    prediction = make_prediction(model, input_data)
    return {"prediction": prediction[0].item()} # item because Fast API does not support numpy.* datatypes
    
@app.post("/update-model")
def update_model(version: str):
    # update the model version
    global model
    print(f"model old params: {model.get_params()}")
    model = load_model("tracking-quickstart", version)
    print(f"model new params: {model.get_params()}")
    return {"model_version": version}

@app.post("/accept-next-model")
def accept_next_model():
    global model_v1, model_v2
    model_v1 = model_v2
    model_v2 = load_model("tracking-quickstart", "2")
    return {"model_version": "2"}