from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from ninept import qwen
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from sklearn.linear_model import BayesianRidge
from scipy.stats import boxcox
import ast

######################################## Lots of Functions ####################################

### impute a dataframe n_imputations many times and return dataframe
def impute_mixed_data(df, n_imputations=1, strategy = "stacking"):
    """
    Imputes missing values in a DataFrame using:
        - MICE with XGBoost for numeric columns
        - Most frequent value for categorical columns.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        n_imputations (int): Number of imputations for multiple imputation. Defaults to 1.
        strategy (str): Strategy for handling multiple imputations:
             - "aggregate": Returns a single DataFrame with aggregated imputations.
            - "stacking": Stacks imputations vertically in a single DataFrame.

    
    Returns:
        pd.DataFrame: The DataFrame with imputed values aggregated across imputations.
    """
    if strategy not in ["aggregate", "stacking"]:
        raise ValueError(f"Invalid strategy '{strategy}'. Choose 'aggregate' or 'stacking'.")

    # Identify columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    ignored_cols = df.columns.difference(numeric_cols.union(categorical_cols))

    # Define imputer. Extreme gradient boosted tree ensembles as a default
    xgb_estimator = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        early_stopping_rounds=None,
        verbosity=0
    )
    if n_imputations == 1:
        numeric_imputer = IterativeImputer(estimator=xgb_estimator, max_iter=100, random_state=42)
    else:   
        #rather use Bayesian Ridge since we need posterior estimation with std estimate when MI is used, not SI
        numeric_imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=100,
        random_state=42,
        sample_posterior=True  # Use posterior sampling
        )
    categorical_imputer = SimpleImputer(strategy="constant") #fill_value=None, add_indicator=FALSE
    #set missing values as their own category seems more sensible
    
    
    # Perform imputations
    imputations = []
    for _ in range(n_imputations):
        df_imputed = df.copy()
        
        # Impute numeric columns
        if not numeric_cols.empty:
            df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        
        # Impute categorical columns
        if not categorical_cols.empty:
            df_imputed[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        
        imputations.append(df_imputed)

    if n_imputations == 1:
        return imputations[0]
    else:
        if(strategy=="aggregate"):
            # Aggregate multiple imputations
            aggregated_df = imputations[0].copy()
            
            # Average numeric columns
            for col in numeric_cols:
                aggregated_df[col] = np.mean(
                    [imputed_df[col].astype(float) for imputed_df in imputations], axis=0
                )
            
            # Mode for categorical columns
            for col in categorical_cols:
                aggregated_df[col] = pd.concat(
                    [imputed_df[col] for imputed_df in imputations], axis=1
                ).mode(axis=1)[0]
            return aggregated_df
        else: #stacking is defeault
            stacking_df = pd.concat(imputations, axis=0, ignore_index=True)
            return stacking_df




def delete_values(df, p):
    """
    Deletes p percent of all values in the DataFrame by replacing them with NaN.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        p (float): Percentage of values to delete (between 0 and 100).

    Returns:
        pd.DataFrame: The DataFrame with missing values introduced.
    """
    if not (0 <= p <= 100):
        raise ValueError("Percentage 'p' must be between 0 and 100.")

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Calculate the number of values to replace
    total_values = df.size
    n_missing = int(total_values * (p / 100))

    # Flatten the DataFrame index for easier random sampling
    flat_indices = [(i, j) for i in range(df.shape[0]) for j in range(df.shape[1])]

    # Randomly select indices to replace with NaN
    missing_indices = np.random.choice(len(flat_indices), n_missing, replace=False)

    # Replace the selected values with NaN
    for idx in missing_indices:
        i, j = flat_indices[idx]
        df.iat[i, j] = np.nan

    return df

def delete_values_with_exclusion(df, p, exclude_column):
    """
    Deletes p percent of all values in the DataFrame by replacing them with NaN,
    while ensuring that the specified column does not get any missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        p (float): Percentage of values to delete (between 0 and 100).
        exclude_column (str): Name of the column to exclude from missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values introduced.
    """
    if not (0 <= p <= 100):
        raise ValueError("Percentage 'p' must be between 0 and 100.")
    if exclude_column not in df.columns:
        raise ValueError(f"Column '{exclude_column}' not found in the DataFrame.")

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Calculate the number of values to replace
    total_values = df.size - len(df[exclude_column])  # Exclude the protected column
    n_missing = int(total_values * (p / 100))

    # Flatten the DataFrame index for easier random sampling
    flat_indices = [
        (i, j) for i in range(df.shape[0]) for j in range(df.shape[1])
        if df.columns[j] != exclude_column
    ]

    # Randomly select indices to replace with NaN
    missing_indices = np.random.choice(len(flat_indices), n_missing, replace=False)

    # Replace the selected values with NaN
    for idx in missing_indices:
        i, j = flat_indices[idx]
        df.iat[i, j] = np.nan

    return df





def train_and_compare(df1, df2, response_variable):
    """
    Trains Random Forest models on two datasets and compares their cross-validated performance.

    Parameters:
        df1 (pd.DataFrame): The first dataset.
        df2 (pd.DataFrame): The second dataset.
        response_variable (str): The name of the response variable (target column).

    Returns:
        None: Prints the performance metrics to the screen.
    """
    # Ensure the response variable exists in both datasets
    if response_variable not in df1.columns or response_variable not in df2.columns:
        raise ValueError(f"Response variable '{response_variable}' not found in both datasets.")

    # Determine if the response is numeric (regression) or categorical (classification)
    is_numeric_response = np.issubdtype(df1[response_variable].dtype, np.number)

    # Split features (X) and response variable (y)
    X1, y1 = df1.drop(columns=[response_variable]), df1[response_variable]
    X2, y2 = df2.drop(columns=[response_variable]), df2[response_variable]

    if is_numeric_response:
        # Regression
        model = RandomForestRegressor(random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)  # Use MSE as the metric
    else:
        # Classification
        model = RandomForestClassifier(random_state=42)
        scorer = make_scorer(accuracy_score)  # Use accuracy as the metric

    # Cross-validate on both datasets
    cv_scores1 = cross_val_score(model, X1, y1, cv=5, scoring=scorer)
    cv_scores2 = cross_val_score(model, X2, y2, cv=5, scoring=scorer)

    if is_numeric_response:
        # Print MSE (negate scores for MSE since we used greater_is_better=False)
        print(f"Dataset 1 - Mean Squared Error: {abs(cv_scores1.mean()):.4f}")
        print(f"Dataset 2 - Mean Squared Error: {abs(cv_scores2.mean()):.4f}")
    else:
        # Print misclassification (1 - accuracy)
        print(f"Dataset 1 - Misclassification Rate: {1 - cv_scores1.mean():.4f}")
        print(f"Dataset 2 - Misclassification Rate: {1 - cv_scores2.mean():.4f}")



def add_missingness_correlation_vars(df, response, threshold):
    """
    Adds missingness indicators for columns with missing values based on correlation with the response variable.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        response (str): The name of the response variable column.
        threshold (float): The correlation threshold to add the missingness indicator.

    Returns:
        pd.DataFrame: The modified DataFrame with new columns for significant missingness indicators.
    """
    if response not in df.columns:
        raise ValueError(f"Response variable '{response}' not found in the DataFrame.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a numeric value.")
    
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Ensure response is numeric (correlation requires numeric data)
    if not np.issubdtype(df[response].dtype, np.number):
        raise ValueError("The response variable must be numeric to calculate correlations.")

    # Identify columns with missing values
    missingness_cols = [col for col in df.columns if col != response and df[col].isnull().any()]

    for col in missingness_cols:
        # Create a missingness indicator (1 if missing, 0 otherwise)
        missing_indicator = df[col].isnull().astype(int)
        # Calculate correlation with the response variable
        correlation = missing_indicator.corr(df[response])
        
        # Add the indicator as a new column if correlation exceeds the threshold
        if abs(correlation) > threshold:
            new_col_name = f"{col}_missing"
            df[new_col_name] = missing_indicator

    return df


# Imagine the response has to be a number
def read(output):
    output = output.strip()
    output.replace(",", ".")
    return int(output)

def call_llm(content, role, tries=10):
    outp = qwen(content, role)
    try:
        return read(outp)
    except:
        if tries == 0:
            raise Exception("Failed to get a valid response from the llm (" + str(outp) + ")")
        else:
            return call_llm(content + f"The last string ('{outp}') was not a valid number. Please answer only with an integer number", role, tries - 1)
        


# Imagine the response has to be an array of integers
def read_mv(output):
    output = output.strip()
    output = output.replace(",", ".")
    # Split the output into parts and try converting each to an integer
    return [int(value) for value in output.split()]

def call_llm_mv(content, role, tries=10):
    outp = qwen(content, role)
    try:
        return read_mv(outp)
    except:
        if tries == 0:
            raise Exception("Failed to get a valid response from the llm (" + str(outp) + ")")
        else:
           # print("This try did not work: " + str(tries))
            return call_llm_mv(
                content + f"The last string ('{outp}') was not a valid array of integers. Please answer only with a space-separated list of integers.",
                role,
                tries - 1
            )

import ast
import re

def read_mv_general(output):
    """
    Tries to parse a string as a list of two integers. If parsing fails,
    it extracts the first two integers from the string.

    Parameters:
        output (str): The input string containing two integers.

    Returns:
        list: A list containing exactly two integers.

    Raises:
        ValueError: If less than two integers are found.
    """
    output = output.strip()
    output = output.replace(",", ".")  # If commas are meant as decimal points

    # Try parsing as a list of integers (e.g., "[1, 2]")
    try:
        parsed = ast.literal_eval(output)
        if isinstance(parsed, list) and len(parsed) == 2 and all(isinstance(x, int) for x in parsed):
            return parsed
    except (ValueError, SyntaxError):
        pass  # Fall back to extracting from string

    # Fallback: Extract integers from the string
    ints = re.findall(r"-?\d+", output)
    ints = [int(x) for x in ints]

    if len(ints) < 2:
        raise ValueError(f"Failed to find at least two integers in: {output}")

    return ints[:2]  # Return the first two integers


def call_llm_mv_2(content, role, tries=25):
    """
    Calls the LLM with retry logic and ensures the response is a valid list of integers.

    Parameters:
        content (str): The input content/query for the LLM.
        role (str): The role for the LLM (e.g., "data science expert").
        tries (int): The maximum number of retry attempts.

    Returns:
        list: A list of integers parsed from the LLM response.

    Raises:
        Exception: If the maximum number of retries is reached without a valid response.
    """
    # Call the LLM
    outp = qwen(content, role)

    try:
        # Attempt to parse the response
        return read_mv_general(outp)
    except Exception as e:
        # Handle failed attempts
        if tries == 0:
            raise Exception(f"Failed to get a valid response from the LLM after multiple attempts. Last response: '{outp}'. Error: {e}")
        else:
            # Update content with feedback and retry
            print(f"Retrying... {tries} attempts left. Last response was invalid: {outp}")
            updated_content = (
                content + 
                f" The last string ('{outp}') was not a valid array of integers. Please answer only with a comma-separated list of integers. Example output: [2, 5]."
            )
            return call_llm_mv_2(updated_content, role, tries - 1)

#### check LLM output for list of lists of integer pairs


def read_pairlist(output):
    """
    Validates and parses the output into a list of lists, where each inner list contains exactly 2 integers.

    Parameters:
        output (str): The input string to validate and parse.

    Returns:
        list of lists: Parsed and validated list of lists containing integers.

    Raises:
        ValueError: If the input is not a valid list of lists with exactly 2 integers each.
    """
    output = output.strip()

    # Parse safely using ast.literal_eval
    try:
        parsed = ast.literal_eval(output)
    except (ValueError, SyntaxError):
        raise ValueError("Input must be a valid Python expression representing a list of lists.")

    # Ensure parsed object is a list of lists
    if not (
        isinstance(parsed, list) and 
        all(
            isinstance(inner, list) and len(inner) == 2 and 
            all(isinstance(i, int) for i in inner)
            for inner in parsed
        )
    ):
        raise ValueError("Input must be a list of lists, where each inner list contains exactly 2 integers.")

    return parsed

def call_llm_pairlist(content, role, tries=10):
    outp = qwen(content, role)
    try:
        return read_pairlist(outp)
    except:
        if tries == 0:
            raise Exception("Failed to get a valid response from the llm (" + str(outp) + ")")
        else:
            print("This try did not work: " + str(tries))
            return call_llm_mv(
                content + f"The last string ('{outp}') was not a valid list of lists, where each inner list contains exactly 2 integers. Please do that.",
                role,
                tries - 1
            )

#function to add dummy missingness to the dataframe for all columns in indices
def add_missingness_columns(df, indices):
    """
    Adds missingness indicators for columns with missing values.
    Ignores categorical columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        indices (list of ints): Column indices to add missingness dummy columns.

    Returns:
        pd.DataFrame: The modified DataFrame with new columns.
    """
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    for col_index in indices:
        if col_index < 0 or col_index >= len(df.columns):
            raise ValueError(f"Column index {col_index} is out of bounds for the DataFrame.")
        
        col_name = df.columns[col_index]

        # Skip categorical columns
        if df[col_name].dtype == "object" or pd.api.types.is_categorical_dtype(df[col_name]):
            print(f"Skipping categorical column: '{col_name}'")
            continue

        # Create a missingness indicator (1 if missing, 0 otherwise)
        missing_indicator = df[col_name].isnull().astype(int)

        # Add the indicator as a new column if it contains both 0 and 1
        if missing_indicator.nunique() > 1:
            new_col_name = f"{col_name}_missing"
            df[new_col_name] = missing_indicator
            print(f"Missingness column has been added: {col_name}")

    return df



def add_power_columns(df, column_indices, power):
    """
    Adds power versions of the specified columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_indices (list): List of column indices to be squared.
        power (int):  e.g. squaring or cubing

    Returns:
        pd.DataFrame: A new DataFrame with additional power columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    for index in column_indices:
        if index < 0 or index >= len(df.columns):
            raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")
        
        
        column_name = df.columns[index]
        # Handle power-specific column naming
        if power == 2:
            new_column_name = f"{column_name}_squared"
        elif power == 3:
            new_column_name = f"{column_name}_cubed"
        else:
            new_column_name = f"{column_name}_power_{power}"
        
        # Ensure the column contains numeric data
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise ValueError(f"Column '{column_name}' is not numeric and cannot be raised to a power.")

        # Add the power-transformed column
        df[new_column_name] = df[column_name] ** power
        print(f"Power column has been added for: {column_name}")

    return df


def add_log_columns(df, column_indices):
    """
    Adds log-transformed versions of the specified columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_indices (list): List of column indices to be log-transformed.

    Returns:
        pd.DataFrame: A new DataFrame with additional log-transformed columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    for index in column_indices:
        if index < 0 or index >= len(df.columns):
            raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")
        
        column_name = df.columns[index]
        
        # Check for non-positive values
        if (df[column_name] <= 0).any():
            print(f"Column '{column_name}' contains non-positive values and will be skipped.")
            continue  # Skip this column

        # Add log-transformed column
        new_column_name = f"{column_name}_log"
        df[new_column_name] = np.log(df[column_name])

        print(f"Log-column has been added for: {column_name}")
    return df


#try to added a list of pairs but this approach failed
def add_interaction_columns(df, column_indices):
    """
    Adds interaction term of the specified columns to the DataFrame.
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_indices (list): List of lists with two integers each.
    Returns:
        pd.DataFrame: A new DataFrame with additional interaction columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    for pair in column_indices:
        if len(pair) != 2:
            raise ValueError("Only consider two way interactions.")
        
        column_name = ""
        for index in pair:
            if index < 0 or index >= len(df.columns):
                raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")

            column_name = column_name + "_" + df.columns[index]
        new_column_name = f"{column_name}_intA"
        df[new_column_name] = df[pair[0]] * df[pair[1]]
        print(f"Interaction column has been added: {new_column_name}")

    return df

def add_interaction_column_pair(df, column_pair):
    """
    Adds an interaction term for a pair of specified columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_pair (list): A list of exactly two integers representing column indices.

    Returns:
        tuple: A tuple containing the updated DataFrame and the pair of columns handled.
    """
    if len(column_pair) != 2:
        raise ValueError("column_pair must contain exactly two integers.")
    
    for index in column_pair:
        if index < 0 or index >= len(df.columns):
            print(column_pair)
            raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")
    
    # Generate the name for the new column
    col1_name = df.columns[column_pair[0]]
    col2_name = df.columns[column_pair[1]]
    new_column_name = f"{col1_name[:5]}*{col2_name[:5]}"

    # Add the interaction column
    df[new_column_name] = df.iloc[:, column_pair[0]] * df.iloc[:, column_pair[1]]

    print(f"Added interaction column '{new_column_name}' as the product of '{col1_name}' and '{col2_name}'.")
    return df, column_pair


def add_boxcox_columns(df, column_indices):
    """
    Adds Box-Cox transformed versions of the specified columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_indices (list): List of column indices to apply the Box-Cox transformation.

    Returns:
        pd.DataFrame: A new DataFrame with additional Box-Cox transformed columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()
    

    for index in column_indices:
        if index < 0 or index >= len(df.columns):
            raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")

        column_name = df.columns[index]
        
        # Check for positive values required for Box-Cox transformation
        if (df[column_name] <= 0).any():
            print(f"Column '{column_name}' contains non-positive values and will be skipped.")
            continue

        # Apply Box-Cox transformation
        transformed_data, lambda_optimal = boxcox(df[column_name])
        new_column_name = f"{column_name}_boxcox"
        df[new_column_name] = transformed_data

        # Print the optimal λ for the transformation
        print(f"Column '{column_name}' transformed with optimal λ = {lambda_optimal:.4f}")

    return df

def enrich_temporal_data(df):
    """
    Identifies temporal columns in the DataFrame, extracts relevant information,
    and adds new columns for weekend status, day of the week, month, season, quarter, and year.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with additional temporal columns.
    """
    df = df.copy()  # Avoid modifying the original DataFrame

    # Check each column for temporal data
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"Processing temporal column: '{col}'")

            # Extract features from the datetime column
            df[f"{col}_is_weekend"] = df[col].dt.dayofweek >= 5  # Saturday (5) or Sunday (6)
            df[f"{col}_day_of_week"] = df[col].dt.day_name()
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_season"] = df[col].dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_year"] = df[col].dt.year

    return df

def determine_imputations(missing_frequency, n):
    """
    Determines the appropriate number of imputations based on missing data frequency and dataset size.
    
    Parameters:
        missing_frequency (float): Proportion of missing values in the dataset (0 to 1).
        n (int): Number of rows in the dataset.

    Returns:
        int: Recommended number of imputations (1, 3, 5, or 10).
        string: Explanation on what and why was performed.
    """
    if missing_frequency < 0.10:
        num_imputations = 1
        reason = "Low missingness, single imputation is sufficient."
    elif missing_frequency < 0.2:
        num_imputations = 3
        reason = "Moderate missingness, a small number of imputations improves stability."
    elif missing_frequency < 0.4:
        num_imputations = 5
        reason = "High missingness, multiple imputations are needed."
    else:
        num_imputations = 10
        reason = "Very high missingness, many imputations are required for robustness."

    # Adjust for dataset size
    if n > 10000 and num_imputations > 3:
        num_imputations = 3
        reason += " Large dataset detected, limiting imputations to 3 to reduce computation time."
    elif n > 50000 and num_imputations > 1:
        num_imputations = 1
        reason += " Very large dataset detected, using single imputation to keep it computationally feasible."

    print(f"Missing frequency: {missing_frequency:.2%}, Dataset size: {n}")
    print(f"Recommended number of imputations: {num_imputations}")
    print(f"Reason: {reason}")

    return num_imputations, reason