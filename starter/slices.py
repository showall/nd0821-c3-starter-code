from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics
from starter.ml.model import inference


def slices(model, cat, X, encoder, lb, cat_features):
    """Computes performance on model slices
    Inputs
    ------
    model : ???
        Trained machine learning model.
    cat : str
        category to be sliced
    X : np.array
        Data used for prediction.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, for processing data.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, for processing data.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    Returns
    -------
    No returns
    """
    with open("slice_output.txt", "a") as f:
        print(cat, file=f)
        for category in cat:
            temp = X[X[category] == X[category][0]]
            X_temp, y_temp, encoder, lb = process_data(
                temp,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(model, X_temp)
            precision, recall, fbeta = compute_model_metrics(y_temp, preds)
            print("#################################", file=f)
            print(category, file=f)
            print("Precision:", precision, file=f)
            print("Recall:", recall, file=f)
            print("F-Beta Score:", fbeta, file=f)
            print("#################################", file=f)

    return
