import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile

# Download Competition Datasets from Kaggle
def download_datasets_from_kaggle(root_folder):
    api = KaggleApi()
    api.authenticate()

    competitions = [
        "digit-recognizer",
        "equity-post-HCT-survival-predictions",
        "home-data-for-ml-course",
        "house-prices-advanced-regression-techniques",
        "spaceship-titanic",
        "store-sales-time-series-forecasting",
    ]

    os.makedirs(root_folder, exist_ok=True)

    for competition in competitions:
        competition_name = competition
        download_path = os.path.join(root_folder, competition_name)
        os.makedirs(download_path, exist_ok=True)

        print(f"\nDownloading required files for: {competition_name}")

        try:
            required_files = ["train.csv", "test.csv", "sample_submission.csv"]

            for file in required_files:
                api.competition_download_file(competition_name, file_name=file, path=download_path)

            print(f"Downloaded required files for {competition_name} to: {download_path}")
        except Exception as e:
            if '403' in str(e):
                print(f"Skipping {competition_name}: You must accept the rules first.")
            else:
                print(f"Error downloading {competition_name}: {e}")

# Unzip the Competition Datasets
def unzip_dataset(root_folder):
    for competition_folder in os.listdir(root_folder):
        competition_path = os.path.join(root_folder, competition_folder)

        if os.path.isdir(competition_path):
            print(f"\nProcessing: {competition_folder}")

            for file_name in os.listdir(competition_path):
                if file_name.endswith('.zip'):
                    zip_file_path = os.path.join(competition_path, file_name)

                    try:
                        print(f"Unzipping {zip_file_path}...")
                        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                            zip_ref.extractall(competition_path)
                        print(f"Unzipped {zip_file_path}")
                        os.remove(zip_file_path)
                    except Exception as e:
                        print(f"Error unzipping {zip_file_path}: {e}")

# Basic Summary regarding all the csv files from Kaggle Competition
def summarize_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"\nSummary for {file_path}:")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print(f"Column Names: {list(df.columns)}")
        print("Missing Values:")
        print(df.isnull().sum())
        print("Basic Statistics:")
        print(df.describe() if df.shape[0] > 0 else "No data available")
        print("-" * 80)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def summarize_all_datasets(root_folder):
    for competition_folder in os.listdir(root_folder):
        competition_path = os.path.join(root_folder, competition_folder)

        if os.path.isdir(competition_path):
            for file_name in os.listdir(competition_path):
                if file_name in ["train.csv", "test.csv", "sample_submission.csv"]:
                    file_path = os.path.join(competition_path, file_name)
                    summarize_csv_file(file_path)

def main():
    kaggle_datasets_folder = "kaggle_datasets"

    os.makedirs(kaggle_datasets_folder, exist_ok=True)

    download_datasets_from_kaggle(kaggle_datasets_folder)

    unzip_dataset(kaggle_datasets_folder)

    summarize_all_datasets(kaggle_datasets_folder)

if __name__ == "__main__":
    main()

