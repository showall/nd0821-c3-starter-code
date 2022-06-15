# Script to train machine learning model.
import os
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
# Add the necessary imports for the starter code.
from ml.data import process_data
import pickle
from ml.model import train_model, compute_model_metrics

#print directory
#sys.path.append('../starter')


# Add code to load in the data.
data = pd.read_csv("data/cleaned_census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",training=False, encoder=encoder, lb=lb)

# Train and save a model.
model = train_model(X_train, y_train)
pred =  model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test,pred)
print(f"precision :{precision} , recall :{recall} , fbeta:{fbeta}")

filename = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'model', "model.pkl"))
with open(filename, 'wb') as file:
    pickle.dump(model, file)

filename = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'model', "encoder.pkl"))
with open(filename, 'wb') as file:
    pickle.dump(encoder, file)

filename = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'model', "lb.pkl"))
with open(filename, 'wb') as file:
    pickle.dump(lb, file)