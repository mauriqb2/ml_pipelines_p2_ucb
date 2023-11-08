# evaluate_model.py
import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import subprocess
import sys
try:
    import seaborn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns
import matplotlib.pyplot as plt
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Logistic Regression model.")
    parser.add_argument("--true_labels_csv", type=str, required=True,
                        help="Path to the csv file with the true labels.")
    parser.add_argument("--predictions_csv", type=str, required=True,
                        help="Path to the csv file with the model predictions.")
    parser.add_argument("--report_csv", type=str, required=True,
                        help="Path where the classification report CSV will be saved as an output.")
    parser.add_argument("--confusion_matrix", type=str, required=True,
                        help="Path where the confusion matrix PNG will be saved as an output.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    mlflow.start_run()

    y_test = pd.read_csv(args.true_labels_csv)['Potability']
    predictions_df = pd.read_csv(args.predictions_csv)
    y_pred = predictions_df['Predicted']

    report = classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"]))

    for key, value in report.items():
        if isinstance(value, dict):
            for metric, score in value.items():
                mlflow.log_metric(f"{key}_{metric}", score)

    df_rep = pd.DataFrame(report).transpose()
    df_rep.to_csv(args.report_csv, index=True)

    mlflow.log_artifact(args.report_csv)

    confusion = confusion_matrix(y_test, y_pred)
    fig_confusion_matrix = plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Potable", "Potable"], yticklabels=["Not Potable", "Potable"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
 
    confusion_matrix_path = f'{args.confusion_matrix}/confusion_matrix.png'
    plt.savefig(confusion_matrix_path)

    mlflow.log_figure(fig_confusion_matrix, 'confusion_matrix.png')
    plt.close(fig_confusion_matrix)
    
    mlflow.end_run()

if __name__ == "__main__":
    main()
