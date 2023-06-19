import mlflow 
import uvicorn
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile

description = """ eazeaze """

tags_metadata = [
    {
        "name": "Machine Learning",
        "description": "Prediction Endpoint."
    }#,
]

app = FastAPI(
    title="WakeUp",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)


@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    Simply returns a welcome message!
    """
    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message

@app.post("/predict", tags=["Machine Learning"])
async def predict(file: UploadFile= File(...)):
    """
    Description à faire 
    """
    response = {"filename": file.filename}
    return response
    """
    # Read data 
    years_experience = pd.DataFrame({"YearsExperience": [predictionFeatures.YearsExperience]})

    # Log model from mlflow 
    logged_model = 'runs:/323c3b4a6a6242b7837681bd5c539b27/salary_estimator'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    prediction = loaded_model.predict(years_experience)

    # Format response
    response = {"prediction": prediction.tolist()[0]}
    """

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)