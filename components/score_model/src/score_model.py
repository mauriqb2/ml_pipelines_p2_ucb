import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
import mlflow

def load_test_data(test_data_path):
    data = pd.read_csv(test_data_path)
    X_test = data.drop(columns=['Potability'])
    y_test = data['Potability']
    return X_test, y_test

def score_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy}")
    return y_pred, accuracy

def save_predictions(predictions, output_path):
    predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
    predictions_df.to_csv(output_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Score a Logistic Regression model.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the csv file containing the test dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved Logistic Regression model.")
    parser.add_argument("--predictions_csv", type=str, required=True, help="Path to save the output predictions csv.")
    return parser.parse_args()

def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def main():
    args = parse_args()

    mlflow.start_run()

    X_test, y_test = load_test_data(args.test_data_path)

    model = load_model(args.model_path)

    predictions, accuracy = score_model(model, X_test, y_test)

    save_predictions(predictions, args.predictions_csv)

    mlflow.log_artifact(args.predictions_csv)

    mlflow.end_run()

if __name__ == "__main__":
    main()
