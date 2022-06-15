import pytest
import pandas as pd


@pytest.fixture
def raw_data():
    """
    Get dataset
    """
    df = pd.read_csv("data/census.csv")
    return df


@pytest.fixture
def cleaned_data():
    """
    Get dataset
    """
    df = pd.read_csv("data/cleaned_census.csv")
    return df


def test_basic_cleaning(raw_data, cleaned_data):
    """
    Check to see if length of data before and after cleaning is the same
    """
    assert len(raw_data) == len(cleaned_data)


def test_len_label(cleaned_data):
    """
    Check to label categories only 2 ranges
    """
    assert len(cleaned_data.salary.unique()) == 2


def test_datatype_features(cleaned_data):
    """
    Check to see features of categories are object
    """
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

    check_cat_features = [
        i
        for i in cleaned_data.columns
        if cleaned_data[i].dtypes == object and i != "salary"
    ]
    assert check_cat_features == cat_features
