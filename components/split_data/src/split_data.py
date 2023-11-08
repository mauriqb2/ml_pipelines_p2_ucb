# split_data.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def save_split_data(X_train, X_test, y_train, y_test, train_dir, test_dir, test_labels_dir):
    train_path = Path(train_dir)
    test_path = Path(test_dir)
    test_labels_path = Path(test_labels_dir)

    X_train.join(y_train).to_csv(train_path, index=False)
    X_test.join(y_test).to_csv(test_path, index=False)
    y_test.to_csv(test_labels_path, index=False)

def split_data(data, test_size, random_state):
    X = data.drop(columns=['Potability'])
    y = data['Potability']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Split dataset into training and testing sets.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the csv file containing the dataset.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Size of the test dataset.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--train_data", type=str, required=True, help="Directory where the train dataset will be saved.")
    parser.add_argument("--test_data", type=str, required=True, help="Directory where the test dataset will be saved.")
    parser.add_argument("--test_labels", type=str, required=True, help="Directory where the test dataset will be saved.")
    return parser.parse_args()

def main():
    args = parse_args()

    mlflow.start_run()

    data = load_data(args.data_path)

    X_train, X_test, y_train, y_test = split_data(data, args.test_size, args.random_state)
    
    save_split_data(X_train, X_test, y_train, y_test, args.train_data, args.test_data, args.test_labels)
    
    mlflow.log_artifacts(args.train_data, "train_data")
    mlflow.log_artifacts(args.test_data, "test_data")
    mlflow.log_artifacts(args.test_labels, "test_labels")

    mlflow.end_run()

if __name__ == "__main__":
    main()
