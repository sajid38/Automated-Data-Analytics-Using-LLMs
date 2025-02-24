import os
os.chdir('/main/ahmed')
from  fe_vince_functions import *
from ninept import qwen

################################################### Main function ##########################################

def vince_feature_engineering(df, 
                            eda_summary = "", #from EDA
                            ext_info = "", #from external knowledge group
                            response = "mpg",
                            print_details = False):  #turn this on for debugging purposes
    
    colnames_string = ", ".join(df.columns) 
    col_num = len(df.columns)
    
    df_new = df.copy()
    #### imputation: First add missing columns for numerical variables, then impute
    # query_missing = ("Have a look at the following columns: " + colnames_string + 
    #                  " . Also consider the results from the explanatory data analysis: " + eda_summary +
    #                  " , these additional information: " + ext_info +  
    #                  " and try to have an educated guess, for which numerical variables the indicator whether the value is missing or not could have predictive power on the response variable: " + response + 
    #                  ". Ignore categorical variables. Return a list of integers and do not output anything else!  If you don't find a useful column, return NULL.")
    # try:
    #     answer_missing = call_llm_mv(query_missing, "data science expert")
    # except Exception:
    #     print("Could not ask LLM for missingness columns.")
    #     answer_missing = ""
    # if print_details:
    #     print(answer_missing)
        
    # try:
    #     df_new = add_missingness_columns(df, answer_missing)
    #     print("Successfully added missingness columns.")
    # except Exception as e:
    #     df_new = df
    #     print(f"Failed to add missing columns: {e}")    
        
    
    # #the number of imputations depend on the size of the dataset and the missingness rate
    # missing_frequency = df.isnull().sum().sum() / df.size
    # n_imputations, explanation = determine_imputations(missing_frequency, df.shape[0])
    # df_new = impute_mixed_data(df_new, n_imputations = n_imputations) #this should never fail
        
    
    #### handle temporal data if the df contains temporal data
    try:
        df_new = enrich_temporal_data(df_new)
        print("Successfully handled temporal data.")
    except Exception as e:
        print(f"Failed to enrich temporal data: {e}")    
    
    
    #### ask LLM about some common transformation
    #query_Ints = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables an interaction term should be added as a new feature which could improve a prediction model on " + response + ", so return a list of two integers and do not output anything else! Example output: [2, 5]"
    query_Squ = ("Have a look at the following columns: " + colnames_string + 
                 " . Also consider the results from the explanatory data analysis: " + eda_summary + 
                 " , these additional information: " + ext_info +  
                 " and try to have an educated guess, for which variables a squared tern should be added as a new feature which could improve a prediction model on " + response +
                 ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [3, 7]")
    
    query_Cub = ("Have a look at the following columns: " + colnames_string + 
                 " . Also consider the results from the explanatory data analysis: " + eda_summary + 
                 " , these additional information: " + ext_info +  
                 " and try to have an educated guess, for which variables a cubic term should be added as a new feature which could improve a prediction model on " + response +
                 ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [1, 4]")
    
    query_Log = ("Have a look at the following columns: " + colnames_string + 
                 " . Also consider the results from the explanatory data analysis: " + eda_summary + 
                 " , these additional information: " + ext_info +  
                 " and try to have an educated guess,  which variable should be log-tranformed  which could improve a prediction model on " + response + 
                 ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [5, 9]")
    
   # query_boxCox = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess,  which variable should be box-cox transformed which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
   # query_temp = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and tell me, which columns contain temporal data, e.g. the date, so return a list of integers and do not output anything else!  If you don't find a column with temporal data, return NULL."
    
    #answer_Ints = call_llm_mv(query_Ints, "data science expert")
    # try:
    #     answer_Squ = call_llm_mv(query_Squ, "data science expert")
    # except Exception:
    #     answer_Squ = list()
    # try:
    #     answer_Cub = call_llm_mv(query_Cub, "data science expert")
    # except Exception:
    #     answer_Cub = list()
    # try:
    #     answer_Log = call_llm_mv(query_Log, "data science expert")
    # except Exception:
    #     answer_Log = list()
        
    # if print_details:
    #     print(answer_Squ)
    #     print(answer_Cub)
    #     print(answer_Log)
    #answer_temp = call_llm_mv(query_temp, "data science expert")
    #answer_boxCox = call_llm_mv(query_boxCox, "data science expert")
     
    
    #Try to perform transformation without crashing the main
    try:
        answer_Squ = call_llm_mv(query_Squ, "data science expert")
        df_new = add_power_columns(df_new, answer_Squ, 2)
        print("Successfully applied squared power columns.")
    except Exception as e:
        print(f"Failed to add squared power columns: {e}")

    try:
        answer_Cub = call_llm_mv(query_Cub, "data science expert")
        df_new = add_power_columns(df_new, answer_Cub, 3)
        print("Successfully applied cubed power columns.")
    except Exception as e:
        print(f"Failed to add cubed power columns: {e}")

    try:
        answer_Log = call_llm_mv(query_Log, "data science expert")
        df_new = add_log_columns(df_new, answer_Log)
        print("Successfully applied log columns.")
    except Exception as e:
        print(f"Failed to add log columns: {e}")

    '''
    try:
        excluded_cols = []
        tmp = add_interaction_column_pair(df_new, answer_Ints)
        df_new = tmp[0]
        excluded_cols.append(tmp[1])
        print("Successfully applied interaction columns.")
    except Exception as e:
        print(f"Failed to add interaction columns: {e}")
    '''
    
    
    
    # Initial query
    query_Ints = (
        "Have a look at the following columns: " + colnames_string +
        " . Also consider the results from the explanatory data analysis: " + eda_summary +
        " , these additional information: " + ext_info +
        " and try to have an educated guess, for which 2 variables an interaction term "
        "should be added as a new feature which could improve a prediction model on " +
        response + ", so return a list of exactly two integers and do not output anything else! Example output: [2, 5]")
       # "Example output: [2, 5]"
    
    excluded_cols = []
    max_iterations = col_num #safety break
    while True:
        if max_iterations == 0:
            break
        max_iterations  = max_iterations - 1
        
        excluded_info = f" The following column pairs have already been excluded: {excluded_cols}."
        
        if print_details:
            print(f"Excluded cols: {excluded_cols}")
        updated_query = query_Ints + excluded_info

        try:
            # Call the LLM to get the next pair of column indices
            answer_Ints = call_llm_mv_2(updated_query, "data science expert")
            if print_details:
                print(f"Number of iterations left: {max_iterations}")
                print(answer_Ints)
            # Validate the LLM output
            if not isinstance(answer_Ints, list) or len(answer_Ints) != 2 or not all(isinstance(i, int) for i in answer_Ints):
                raise ValueError(f"Invalid response from LLM:  Expected a list of two integers.")

            # Try to add the interaction column
            try:
                tmp = add_interaction_column_pair(df_new, answer_Ints)
                df_new = tmp[0]
                new_excluded_cols = tmp[1]
                #excluded_cols.append(tmp[1])  # Update excluded_cols with the handled pair
                #excluded_cols.extend(tmp[1] if isinstance(tmp[1], list) else [tmp[1]])  # Ensure list format
                #excluded_cols = [col for pair in excluded_cols for col in pair] #flatten
                
                # extend
                excluded_cols.extend(new_excluded_cols if isinstance(new_excluded_cols, list) else [new_excluded_cols])
                excluded_cols = [col for pair in excluded_cols for col in (pair if isinstance(pair, (list, tuple)) else [pair])] #flattten
                
                print(f"Successfully applied interaction columns for pair: {new_excluded_cols}")
            except ValueError as e:
                print(f"Could not add Interaction column: {e}")
                
        except ValueError as ve:
            print(f"Validation error: {ve}")
            break  # Exit if the LLM response is invalid

        except Exception as e:
            print(f"Failed to add interaction columns: {e}")
            break  # Exit on other exceptions
    
    #delete duplicate columns, this can sometimes happen unfortunately
    df_new = df_new.loc[:, ~df_new.T.duplicated()]
    
    #Ask LLM which transformations have been performed
    new_colnames =  ", ".join(df_new.columns)
    
    # query_trafos = "Which feature engineering transformations have been done?"
    # "Look at the column names" + new_colnames + "and have an educated guess "
    # " based on the column name endings: "
    # "the column names ending e.g. in _squared when the orginal column was added squared, _log for a "
    # "log-transformation, _missing for adding a dummy encoded column indicating if the observation has "
    # "a missing value in the orginal variable. _is_weekend, _day_of_week, etc, are indicators that"
    # "date data was enriched with furhter information. _intA indicate an added interaction term. "
    
    # answer_trafos = qwen(query_trafos)
    # if print_details:
    #     print(answer_trafos + "\n")

    query_trafos = ("We have performed a few feature engineering transformations, as indicated by the column names ending e.g. " +
                    "in _Squ when the orginal column was added squared, _log for a logtransformation etc. Compare the orginal column names: " + colnames_string + 
                    " with the new column names: " + new_colnames + "and describe the performed transformation very briefly!")
    answer_trafos = qwen(query_trafos)
    
    if print_details:
        print(answer_trafos + "\n")
    
    #return transformed dataframe and a description of performed transformations
    results = {
    "transformed data": df_new,
    "explanation": answer_trafos
    }
    return results


def imputation_by_LLM(df, 
                            eda_summary = "", #from EDA
                            ext_info = "", #from external knowledge group
                            response = "mpg",
                            print_details = False):
    df_new = df.copy()
    
    colnames_string = ", ".join(df_new.columns) 
    
        #### imputation: First add missing columns for numerical variables, then impute
    query_missing = ("Have a look at the following columns: " + colnames_string + 
                     " . Also consider the results from the explanatory data analysis: " + eda_summary +
                     " , these additional information: " + ext_info +  
                     " and try to have an educated guess, for which numerical variables the indicator whether the value is missing or not could have predictive power on the response variable: " + response + 
                     ". Ignore categorical variables. Return a list of integers and do not output anything else!  If you don't find a useful column, return NULL.")
    try:
        answer_missing = call_llm_mv(query_missing, "data science expert")
    except Exception:
        print("Could not ask LLM for missingness columns.")
        answer_missing = ""
    if print_details:
        print(answer_missing)
        
    try:
        df_new = add_missingness_columns(df_new, answer_missing)
        print("Successfully added missingness columns.")
    except Exception as e:
        print(f"Failed to add missing columns: {e}")    
        
    
    #the number of imputations depend on the size of the dataset and the missingness rate
    missing_frequency = df.isnull().sum().sum() / df.size
    n_imputations, explanation = determine_imputations(missing_frequency, df.shape[0])
    
    try:
        df_new = impute_mixed_data(df_new, n_imputations = n_imputations) #this should never fail
    except:
        df_new = df
    return df_new


