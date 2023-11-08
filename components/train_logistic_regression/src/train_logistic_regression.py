# train_logistic_regression.py
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

def load_data(train_data_path):
    data = pd.read_csv(train_data_path)
    X_train = data.drop(columns=['Potability'])
    y_train = data['Potability'] 
    return X_train, y_train

def train_model(X_train, y_train, model_output_path):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    with open(model_output_path, 'wb') as file:
        pickle.dump(model, file)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a logistic regression model.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data csv file.")
    parser.add_argument("--trained_model", type=str, required=True, help="Path to save the trained model.")
    return parser.parse_args()

def main():
    args = parse_args()
    X_train, y_train = load_data(args.train_data_path)
    train_model(X_train, y_train, args.trained_model)

if __name__ == "__main__":
    main()
