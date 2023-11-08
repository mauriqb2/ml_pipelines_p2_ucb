import subprocess
import sys
import mlflow

try:
    import seaborn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def save_plots(data, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig_hist = plt.figure()
    plt.hist(data['Potability'])
    plt.title('Potability Histogram')
    hist_path = f'{output_dir}/potability_histogram.png'
    plt.savefig(hist_path)
    mlflow.log_figure(fig_hist, 'potability_histogram.png')
    plt.close(fig_hist)

    fig_heatmap = plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    heatmap_path = f'{output_dir}/correlation_heatmap.png'
    plt.savefig(heatmap_path)
    mlflow.log_figure(fig_heatmap, 'correlation_heatmap.png')
    plt.close(fig_heatmap)

    pair_plot = sns.pairplot(data, hue='Potability', diag_kind='kde')
    plt.suptitle('Pair Plot')
    plt.subplots_adjust(top=0.9)
    pairplot_path = f'{output_dir}/pair_plot.png'
    pair_plot.savefig(pairplot_path)
    mlflow.log_figure(pair_plot.fig, 'pair_plot.png')
    plt.close(pair_plot.fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Data exploration script.")
    parser.add_argument("--exploration_data", type=str, help="Directory containing the data_cleaned.csv file for data exploration.")
    parser.add_argument("--exploration_output", type=str, help="Directory where the plots will be saved.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    mlflow.start_run()

    data = load_data(args.exploration_data)
    save_plots(data, args.exploration_output)
    
    mlflow.end_run()

if __name__ == "__main__":
    main()
