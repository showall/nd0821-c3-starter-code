from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting":
                               "Hello, thank you for grading my project"}


def test_predict_gt50k():
    r = client.post(
        "/predict",
        json={
            "age": 40,
            "workclass": "Private",
            "fnlgt": 193524,
            "education": "Doctorate",
            "education-num": 16,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "United-States",
        },
    )

    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_predict_lt50k():
    r = client.post(
        "/predict",
        json={
            "age": 38,
            "workclass": "Private",
            "fnlgt": 215646,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Divorced",
            "occupation": "Handlers-cleaners",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )

    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}
