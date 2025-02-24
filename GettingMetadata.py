import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile

# Getting Metadata information regarding Kaggle Competitions
def get_Metadata(competition_name, file_path):
    try:
        api = KaggleApi()
        api.authenticate()

        competitions = api.competitions_list()
        
        with open(file_path, "a", encoding="utf-8") as file:
            for comp in competitions:
                if comp.ref == competition_name:
                    file.write("Competition Details")
                    file.write(f" Title: {comp.title}")
                    file.write(f" Description: {comp.description}")   
                    file.write(f" URL: {comp.ref}")
                    file.write(f" Reward: {comp.reward}")
                    file.write(f" Category: {comp.category}")
                    file.write(f" Tags: {comp.tags}")
                    file.write(f" Evaluation Metric: {comp.evaluationMetric}\n")
                    return

            file.write(f"Competition '{competition_name}' not found.\n")
    
    except Exception as e:
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(f"Error fetching competition details for '{competition_name}': {e}\n")

if __name__ == "__main__":
    output_file = "kaggle_competitions_details.txt"
    
    if os.path.exists(output_file):
        os.remove(output_file)

    competitions = [
        "https://www.kaggle.com/competitions/digit-recognizer",
        "https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions",
        "https://www.kaggle.com/competitions/home-data-for-ml-course",
        "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
        "https://www.kaggle.com/competitions/spaceship-titanic",
        "https://www.kaggle.com/competitions/store-sales-time-series-forecasting"
    ]

    for competition in competitions:
        get_Metadata(competition, output_file)

    print(f"Competition details saved to {output_file}")

