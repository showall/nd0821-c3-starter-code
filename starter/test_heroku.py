"""
Heroku test script
"""
import requests


def check_heroku():
    data = {
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
    }
    r = requests.post("https://nddemoprojectapp.herokuapp.com/predict/",
                      json=data)
    print(r.status_code)
    print(r.json())
    # assert r.status_code == 200


if __name__ == "__main__":
    check_heroku()
