"""
UCLA Admission Prediction - MLP training pipeline
(converted from notebooks/ucla_admission_model.ipynb, bugs fixed)

Fixes vs. the original notebook:
- `del MLP` appeared BEFORE the model was ever created -> NameError on a fresh
  run. Removed.
- `accuracy_score` was used two cells before it was imported -> imports are now
  all at the top.
- Comment said "standard scaler" while the code used MinMaxScaler -> corrected.
- The notebook never saved the scaler/model, yet app.py depends on
  ucla_scaler.pkl and ucla_mlp_model.pkl -> this script now produces both.
- Scaler is fitted on a DataFrame so `feature_names_in_` is preserved, which
  the Streamlit app relies on for column order.
- Paths resolved relative to this file; plots saved instead of only shown.
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "Admission.csv")
SCALER_PATH = os.path.join(HERE, "models", "ucla_scaler.pkl")
MODEL_PATH = os.path.join(HERE, "models", "ucla_mlp_model.pkl")
RANDOM_STATE = 123


def load_and_prepare():
    data = pd.read_csv(DATA_PATH)

    # binary target: strong admission chance (>= 0.8) vs not
    data["Admit_Chance"] = (data["Admit_Chance"] >= 0.8).astype(int)
    data = data.drop(["Serial_No"], axis=1)

    # one-hot encode the categorical columns
    # (target stays numeric - MLPClassifier expects numeric classes)
    data["University_Rating"] = data["University_Rating"].astype("object")
    data["Research"] = data["Research"].astype("object")
    clean = pd.get_dummies(data, columns=["University_Rating", "Research"],
                           dtype="int")

    X = clean.drop(["Admit_Chance"], axis=1)
    y = clean["Admit_Chance"]
    return X, y


def main():
    X, y = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    # min-max scale features to [0, 1]; fit on the TRAIN split only.
    # Fitting on DataFrames keeps feature_names_in_, which app.py needs.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(3, 3), batch_size=50,
                        max_iter=200, random_state=RANDOM_STATE)
    mlp.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, mlp.predict(X_train_scaled))
    y_pred = mlp.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred))

    # loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(mlp.loss_curve_, label="Loss", color="blue")
    plt.title("MLP Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(HERE, "assets", "loss_curve.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # save artifacts used by the Streamlit app
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(mlp, f)
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Saved model  -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
