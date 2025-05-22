# src/train.py
import pandas as pd
from model3 import PatientRiskModel2
import pickle
import os


def train_model():
    # Read training data
    df = pd.read_csv('C:/Users/Swathi/OneDrive/Documents/Desktop/patient risk/PatientRecords.csv')

    # Initialize and train model
    model = PatientRiskModel2()
    model.train(df)

    # Ensure model directory exists
    model_path = 'C:/Users/Swathi/OneDrive/Documents/Desktop/patient risk/models'
    os.makedirs(model_path, exist_ok=True)

    # Save trained model
    with open(os.path.join(model_path, 'risk_model3.pkl'), 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
