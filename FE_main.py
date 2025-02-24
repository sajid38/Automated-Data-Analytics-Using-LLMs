#### This is the function supposed to be called in the overall pipeline ######
import os
os.chdir('/main/ahmed')
import pandas
from sklearn.preprocessing import StandardScaler
from  fe_vince_main import *
import feature_generation #tim
#1 call imputation
#2 call vince FE approach
#3 call Tim FE approach
#4 aggregate and return

##### info for the preprocessing group:
# if the data contains temporal data, e.g. the date, it should be in datetime64_any_dtype format.
# See enrich_temporal_data() function in fe_vince_functions.py for more details

#fe_main performs feature engineering
def fe_main(df, eda_summary, ext_info, response, apply_standardization=True, print_details = False): 
    """
    Main function to apply feature engineering to a dataset.

    This function performs both standard (hardcoded) and flexible feature engineering steps,
    including data imputation and feature generation. It integrates multiple components of
    the feature engineering pipeline to produce a transformed dataset ready for modeling.

    Parameters:
        df (pd.DataFrame): 
            The input dataset to be processed. This is typically raw data requiring cleaning 
            and transformation.
        eda_summary (str): 
            A summary of exploratory data analysis (EDA) results. Includes insights such as 
            missing value statistics or other metadata derived from EDA.
        ext_info (str): 
            External information that supplements the dataset, such as domain knowledge or 
            additional contextual details.
        response (str): 
            The name of the target variable in the dataset. Used during transformations to ensure
            relevance for predictive modeling tasks.
        apply_standardization (bool): 
            If the features of the dataset should be standardized in the last step.
            This means, to transform every variable to have mean 0 and variance 1.

    Returns:
        list: A list containing three elements:
              1. df_new (pd.DataFrame): The transformed dataset after applying all feature engineering steps.
              2. trafos_summary (str): A summary of transformations applied during standard
                 feature engineering steps (e.g., imputations).
              3. generation_summary (str): Metadata about newly generated features during flexible
                 feature engineering steps.
    """
    
    # Only apply feature engineering if the name of the response variable is found
    if response not in df.columns:
        print(f"Response variable '{response}' not found in the dataset.")
        return list(df, 
                    "No feature engineering was made, because the response variable was not found.", 
                    "No feature generation was made, because the response variable was not found.")

    # If the description is too long, shorten it
    if len(eda_summary) > 2000:
        eda_summary = eda_summary[:2000] + "...\n"
        print("Unfortunately, the EDA summary is too large and had to be shortened...")
    if len(ext_info) > 2000:
        ext_info = ext_info[:2000] + "...\n"
        print("Unfortunately, the external information string is too large and had to be shortened...")
        
    print("Performing imputation")
    df_imp = imputation_by_LLM(df, eda_summary = eda_summary, ext_info = ext_info, response=response)
    
    print("Performing imputation and hard coded standard feature engineering steps.\n")
    fe_vince_results = vince_feature_engineering(df_imp, eda_summary, ext_info, response, print_details=print_details) #including imputation
    df_vince = fe_vince_results["transformed data"]
    trafos_summary = fe_vince_results["explanation"]
    
    print("Performing flexible feature engineering steps.\n")
    #df_new, generation_summary = feature_generation.feature_generation(df_new, eda_summary, ext_info, response)
    df_tim, generation_summary = feature_generation.feature_generation(df_imp, eda_summary, ext_info, response) #rather do the FE seperately
    
    df_new = pd.concat([df_vince, df_tim], axis=1)
    df_new = df_new.loc[:, ~df_new.T.duplicated()] #remove duplicates
    
    if apply_standardization:
        try:
            print("Performing standardization.\n")
            scaler = StandardScaler()
            features_to_standardize = df_new.drop(columns=[response])  # Exclude the response variable if necessary
            df_standardized = pd.DataFrame(scaler.fit_transform(features_to_standardize), columns=features_to_standardize.columns)
            # If you want to keep the response variable in the standardized DataFrame:
            df_standardized[response] = df_new[response].values
            df_new = df_standardized
        except:
            print("There was an error in applying Standardization.")

        results = {
            "df_new": df_new,
            "fe_summary": trafos_summary + generation_summary
        }
    return results

