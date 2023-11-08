import argparse
import pandas as pd
from pathlib import Path

def clean_data(input_csv_path, cleaned_data_output):
    data = pd.read_csv(input_csv_path)

    data.fillna(data.mean(), inplace=True)

    cleaned_csv_path = Path(cleaned_data_output)
    data.to_csv(cleaned_csv_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Data cleaning script.")
    parser.add_argument("--clean_data", type=str, help="Path of the CSV file to clean.")
    parser.add_argument("--cleaned_data", type=str, help="Path to save the cleaned CSV file.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    clean_data(args.clean_data, args.cleaned_data)

if __name__ == "__main__":
    main()
