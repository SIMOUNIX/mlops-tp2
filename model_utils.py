import mlflow.sklearn

def load_model(name: str, version: str):
    mlflow.set_tracking_uri(uri="http://localhost:8080")
    model_uri = f"models:/{name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction