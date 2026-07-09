# UCLA Admission Prediction (Neural Network)

Predicts a graduate applicant's chance of admission to UCLA using a Multi-Layer Perceptron (MLP) neural network trained on academic profile data, served through a Streamlit app.

## Business Question

Which applicant attributes (GRE, TOEFL, CGPA, research experience, recommendation strength) best predict a strong chance of admission, and can we estimate that chance for a new applicant?

## Results

| Metric | Score |
|--------|-------|
| Train accuracy | 92.5% |
| **Test accuracy** | **90.0%** |

Target: strong admission chance (probability ≥ 0.8) vs. not. Model: MLP with two hidden layers of 3 neurons, min-max scaled features.

## Project Structure

```
ucla-admission-prediction/
├── app.py                     # Streamlit web app
├── train_model.py             # Training pipeline (saves scaler + model)
├── data/
│   └── Admission.csv          # Dataset (500 applicants)
├── models/
│   ├── ucla_mlp_model.pkl     # Trained MLPClassifier
│   └── ucla_scaler.pkl        # Fitted MinMaxScaler
├── assets/
│   └── loss_curve.png         # Training loss curve
├── notebooks/
│   └── ucla_admission_model.ipynb
├── requirements.txt
└── README.md
```

## Features

GRE score, TOEFL score, university rating (one-hot), SOP strength, LOR strength, CGPA, research experience (one-hot)

## How to Run

```bash
pip install -r requirements.txt
python train_model.py       # retrain: saves models/ pickles + loss curve
streamlit run app.py        # launch the prediction app
```

## Tech Stack

Python, pandas, scikit-learn (MLPClassifier, MinMaxScaler), matplotlib, Streamlit

## Disclaimer

Educational demo — not intended for real admissions decisions.
