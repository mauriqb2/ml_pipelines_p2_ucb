# train_decission_tree.py
import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pickle
import mlflow
import matplotlib.pyplot as plt

def load_data(train_data_path):
    data = pd.read_csv(train_data_path)
    X_train = data.drop(columns=['Potability'])
    y_train = data['Potability']
    return X_train, y_train

def train_model(X_train, y_train, model_output_path):
    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=3, max_depth=4)
    dt.fit(X_train, y_train)
    feature_names = X_train.columns.to_list()

    fig = plt.figure(figsize=(25,20))
    _ = plot_tree(dt, 
                  feature_names=feature_names,  
                  class_names=['Not Potable', 'Potable'],
                  filled=True)

    mlflow.log_figure(fig, 'decission_tree_figure.png')

    with open(model_output_path, 'wb') as file:
        pickle.dump(dt, file)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a decision tree model.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data csv file.")
    parser.add_argument("--trained_model", type=str, required=True, help="Path to save the trained model.")
    return parser.parse_args()

def main():
    args = parse_args()
    X_train, y_train = load_data(args.train_data_path)
    train_model(X_train, y_train, args.trained_model)

if __name__ == "__main__":
    main()
