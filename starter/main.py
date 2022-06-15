# Put the code for your API here.
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference
from starter.ml.data import process_data
import pickle
import pandas as pd


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -f") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# with open("./model/model.pkl", "rb") as f:
#      model = pickle.load(f)
model = pickle.load(open("./model/model.pkl", "rb"))

with open("./model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("./model/lb.pkl", "rb") as f:
    lb = pickle.load(f)

app = FastAPI()


class employee(BaseModel):
    age: int = Field(example=54)
    workclass: str = Field(example="Private")
    fnlgt: int = Field(example=90210)
    education: str = Field(example="Masters")
    education_num: int = Field(alias="education-num", example=14)
    marital_status: str = Field(alias="marital-status", example="Divorced")
    occupation: str = Field(example="Exec-managerial")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Female")
    capital_gain: int = Field(alias="capital-gain", example=0)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country",
                                example="United-States")

    class Config:
        allow_population_by_field_name = True


@app.get("/")
async def welcome_message():
    return {"greeting": "Hello, thank you for grading my project"}


@app.post("/predict")
async def predict(person: employee):
    array = [
        [
            person.age,
            person.workclass,
            person.fnlgt,
            person.education,
            person.education_num,
            person.marital_status,
            person.occupation,
            person.relationship,
            person.race,
            person.sex,
            person.capital_gain,
            person.capital_loss,
            person.hours_per_week,
            person.native_country,
        ]
    ]

    df = pd.DataFrame(
        data=array,
        columns=[
            "age",
            "workclass",
            "fnlgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
        ],
    )

    processed_df, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    prediction = inference(model, processed_df)
    prediction = lb.inverse_transform(prediction)[0]
    return {"prediction": prediction}
