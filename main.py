from ninept import qwen
import subprocess
import os
os.chdir('/main/ahmed')
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sklearn.preprocessing import StandardScaler
from  fe_vince_functions import *
from  FE_main import *
import feature_generation #tim
import json
import PyPDF2
# Run all the other python files as a process

#Data part
def run_file(filename):
    try:
        print(f"\nRunning {filename}...")
        subprocess.run(["python", filename], check=True)
        print(f"{filename} executed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {filename}: {e}")

def llm_evaluation(file_path, dataset_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if dataset_name.lower() in line.lower():
                return line

    return "Dataset description not found."
#eda part
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def automated_eda(df, unique_threshold=20):
    description = ""  # Initialize the description string
    
    # Basic Information
    description += "--- Basic Information ---\n"
    description += f"Number of Rows: {df.shape[0]}\n"
    description += f"Number of Columns: {df.shape[1]}\n"
    description += "\nColumn Names:\n"
    description += ", ".join(df.columns.tolist()) + "\n\n"

    # Data Types
    description += "--- Data Types ---\n"
    description += df.dtypes.to_string() + "\n\n"

    # Missing Values Analysis
    description += "\nMissing Values:\n"
    missing_vals = df.isnull().sum()
    missing_data = missing_vals[missing_vals > 0]
    description += str(missing_data) + "\n"
    description += "\nPercentage of Missing Values:\n"
    description += str((missing_vals / len(df)) * 100) + "\n"
    
    # Drop high-uniqueness categorical columns
    high_uniqueness_cols = []
    for col in df.select_dtypes(include=[object]).columns:
        if df[col].nunique() > unique_threshold:
            high_uniqueness_cols.append(col)
    df = df.drop(columns=high_uniqueness_cols)
    description += f"\nDropped High-Uniqueness Columns: {high_uniqueness_cols}\n"

    # Constant Columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    description += "--- Constant Columns ---\n"
    if constant_columns:
        description += f"The dataset contains {len(constant_columns)} constant column(s): {', '.join(constant_columns)}\n"
    else:
        description += "No constant columns detected.\n\n"

    # Duplicate Rows
    duplicate_rows = df.duplicated().sum()
    description += "--- Duplicate Rows ---\n"
    description += f"Number of duplicate rows: {duplicate_rows}\n\n"

    # Distribution Analysis for Numeric Columns
    description += "\nSkewness and Kurtosis of Numeric Features:\n"
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skew_kurt = pd.DataFrame({
        "Skewness": df[numeric_cols].apply(skew),
        "Kurtosis": df[numeric_cols].apply(kurtosis)
    })
    description += str(skew_kurt) + "\n"
    
    # Boxplot Summary Information
    description += "\nBoxplot Summary Information (Min, Q1, Median, Q3, Max):\n"
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        min_val = df[col].min()
        max_val = df[col].max()
        median = df[col].median()
        description += f"{col} - Min: {min_val}, Q1: {q1}, Median: {median}, Q3: {q3}, Max: {max_val}\n"
        
        # Outliers based on IQR method
        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
        description += f"  Outliers in {col}: {len(outliers)}\n"
    
    # Correlation Analysis on numeric data only
    correlation = df.select_dtypes(include=[np.number]).corr()
    description += "\nCorrelation Analysis (Numeric Feature Pairs):\n"
    for i in range(len(correlation.columns)):
        for j in range(i):
            description += f"{correlation.columns[i]} and {correlation.columns[j]} - Correlation: {correlation.iloc[i, j]:.2f}\n"
    
    # Categorical Columns Analysis
    categorical_cols = df.select_dtypes(include=[object]).columns
    description += "\n--- Categorical Columns ---\n"
    description += f"Categorical Columns: {categorical_cols.tolist()}\n"
    
    description += "\n--- Count Analysis of Categorical Features ---\n"
    label_encoder = LabelEncoder()
    one_hot_encoded_cols = []
    new_df = df.copy()
    
    for col in categorical_cols:
        unique_values = df[col].nunique()
        most_frequent_value = df[col].mode()[0]
        most_frequent_count = df[col].value_counts().iloc[0]
        description += f"{col} - Unique Values: {unique_values}, Most Frequent Value: {most_frequent_value} ({most_frequent_count} occurrences)\n"
        
        # Encoding Categorical Variables
        if unique_values == 2:
            # Label Encoding for Binary Categorical Variables
            new_df[col] = label_encoder.fit_transform(df[col])
            description += f"  {col} - Applied Label Encoding\n"
        elif unique_values > 2:
            # One-Hot Encoding for Multi-Class Categorical Variables
            one_hot_encoded = pd.get_dummies(df[col], prefix=col)
            new_df = new_df.drop(columns=[col]).join(one_hot_encoded)
            one_hot_encoded_cols.extend(one_hot_encoded.columns.tolist())
            description += f"  {col} - Applied One-Hot Encoding\n"
    
    description += "\n--- Final Processed Data ---\n"
    description += f"Shape of Final Processed Data: {new_df.shape}\n"
    description += f"Columns after encoding: {new_df.columns.tolist()}\n"
    
    return description, new_df

def generate_eda_with_qwen(query):
    try:
        prompt = f"I am providing you summary of exploratory data analysis result of a training dataset. Can you provide me some meaningful information within 2400 characters? Here is the summary:\n\n{query}"
        summary = qwen(content=prompt, role="You are a data science expert.")
        
        return summary.strip()
    
    except Exception as e:
        print(f"Error summarizing content with Qwen: {e}")
        return "An error occurred during summarization"
#EK
def generate_queries_with_qwen(query):
    try:
        prompt = f"Generate multiple variations of this search query that cover related topics:\n\n{query}"
        generated_queries = qwen(content=prompt, role="You are a helpful assistant who generates diverse search queries.")
        
        # Split by lines or commas to handle multiple outputs from Qwen
        return [q.strip() for q in generated_queries.split("\n") if q.strip()]
    
    except Exception as e:
        print(f"Error generating queries with Qwen: {e}")
        return [query]  # Fallback to original query if generation fails

# Function: Perform Web Search Using DuckDuckGo (for Multiple Queries)
def perform_duckduckgo_search_multiple(queries):
    try:
        ddgs = DDGS()
        aggregated_results = []
        
        for query in queries:
            print(f"\nSearching with query: {query}")
            
            results = []
            for result in ddgs.text(query, max_results=5):  # Fetch top 5 results per query
                results.append((result['href'], result['title']))
            
            aggregated_results.extend(results)
        
        # Remove duplicate URLs by converting to a dictionary (preserves order)
        unique_results = list({url: title for url, title in aggregated_results}.items())
        
        return unique_results
    
    except Exception as e:
        print(f"Error performing DuckDuckGo search: {e}")
        return []

# Function: Scrape Web Content from URLs (Improved with Headers)
def scrape_web_content(urls):
    contents = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    for url, title in urls[:5]:  # Process up to top 5 URLs only
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise error if request fails
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract main content (simplified extraction logic)
            paragraphs = soup.find_all('p')
            text_content = " ".join([p.get_text() for p in paragraphs])
            
            contents.append((title, url, text_content))
        
        except Exception as e:
            print(f"Error scraping URL ({url}): {e}")
    
    return contents

# Function: Summarize Content Using Qwen
def summarize_with_qwen(contents):
    try:
        combined_text = "\n\n".join([f"Source Title: {title}\nSource URL: {url}\n{text}" 
                                     for title, url, text in contents])
        
        prompt = f"Please summarize the following information concisely while citing sources:\n\n{combined_text}"
        
        summary = qwen(content=prompt.strip(), role="You are a helpful assistant who summarizes information.")
        
        return summary.strip()
    
    except Exception as e:
        print(f"Error summarizing content with Qwen: {e}")
        return "An error occurred during summarization."


#model selection
# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return None

# Function to extract Python code from Qwen's response
def extract_model_code(response):
    """
    Extract Python code from Qwen's response using triple backticks.
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    return "\n".join(matches).strip() if matches else None

# Function to query Qwen for analysis
def query_qwen_for_analysis(prompt, context=""):
    """
    Query Qwen for a Python script based on the given prompt and context.
    """
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    strict_prompt = (
        "Provide a complete and executable Python script for the given request. "
        "Ensure that all necessary imports, functions, and logic are included. "
        "DO NOT use an external dataset URL. Use the provided preprocessed data. "
        "Do NOT truncate responses. If the output is long, return it in sequential blocks using triple backticks ```python```."
    )
    response = qwen(content=full_prompt + "\n\n" + strict_prompt)
    return response

# Function to identify the best model using Qwen and the PDF guide
def identify_best_model(X_train, y_train, pdf_context):
    """
    Ask Qwen to identify the best model based on the provided data, target, and PDF guide.
    """
    prompt = (
        "Based on the following dataset and the provided model selection guide, identify the best machine learning model for the given target. "
        "The dataset has the following shapes:\n"
        f"- X_train: {X_train.shape}\n"
        f"- y_train: {y_train.shape}\n"
        "The target is a binary classification problem. "
        "Choose the best model from the following candidates: Logistic Regression, Decision Tree, Random Forest. "
        "Return only the name of the best model (e.g., 'Random Forest')."
    )
    response = qwen(content=pdf_context + "\n\n" + prompt)
    return response.strip()

# Function to generate script for the best model
def generate_best_model_script(best_model_name, X_train, y_train, pdf_context):
    """
    Ask Qwen to generate a script for the best model.
    """
    prompt = (
        f"Generate a complete and executable Python script for the best model ({best_model_name}). "
        "The script should include the following steps:\n"
        "1. Load the preprocessed data (X_train, y_train).\n"
        "2. Initialize the best model with default or recommended hyperparameters.\n"
        "3. Fit the model on the training data (X_train, y_train).\n"
        "4. Save the trained model to a file (e.g., using `joblib` or `pickle`).\n"
        "5. Ensure the script is complete, executable, and properly formatted.\n"
        "6. Use consistent indentation (4 spaces per level).\n"
        "7. Ensure all brackets (`[`, `{`, `(`) are properly closed.\n"
        "8. Ensure the script is properly formatted and free of syntax errors.\n"
        "9. Avoid unexpected indentation or misaligned code blocks.\n"
        "10. Generate complete and executable code without truncation or omissions.\n"
        "11. Strictly enforce proper indentation (4 spaces per level) for all code blocks.\n"
        "12. Ensure all nested structures (e.g., lists, dictionaries, loops) are properly indented.\n"
        "DO NOT use an external dataset URL. Use the provided preprocessed data."
    )
    response = query_qwen_for_analysis(prompt, context=pdf_context)
    return extract_model_code(response)

# Function to save the generated script to a JSON file
def save_script_to_json(script_content, json_file_path):
    """
    Save the generated script to a JSON file.
    """
    script_data = {"script": script_content}
    with open(json_file_path, "w") as file:
        json.dump(script_data, file, indent=4)
    print(f"✅ Script saved to JSON file: {json_file_path}")



def llm_output():
    datasets = [
    "digit-recognizer",
    "equity-post-HCT-survival-predictions",
    "home-data-for-ml-course",
    "house-prices-advanced-regression-techniques",
    "spaceship-titanic",
    "store-sales-time-series-forecasting"
    ]

    print("Available datasets:")
    for dataset in datasets:
        print(f"- {dataset}")

    selected_dataset = input("\nEnter the dataset name: ").strip()

    file_path = "kaggle_competitions_details.txt"
    result = llm_evaluation(file_path, selected_dataset)
    print("Hello! This is your QWEN LLM Agent !!!")
    #print(result)
    dataset_path = f"/main/ahmed/kaggle_datasets/{selected_dataset}"
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    train_file = next((f for f in csv_files if 'train' in f.lower()), None)
    test_file = next((f for f in csv_files if 'test' in f.lower()), None)
    if train_file and test_file:
        train_df = pd.read_csv(os.path.join(dataset_path, train_file))
        test_df = pd.read_csv(os.path.join(dataset_path, test_file))
        target_variable = train_df.columns[-1] if not train_df.empty else None
        train_description, process_train_df = automated_eda(train_df)
        print(train_description)
        print(process_train_df.head())
        response_train = generate_eda_with_qwen(train_description)
        print(response_train)
        print("Welcome! Please enter your query:")
        user_query = result.strip()  # Step 1: Get User Input
        print("\nGenerating multiple related queries...")
        
        generated_queries = generate_queries_with_qwen(user_query)  # Step 2: Generate Multiple Queries
        
        print(f"\nGenerated Queries:\n{generated_queries}")
        
        print("\nSearching the web...")
        
        search_results = perform_duckduckgo_search_multiple(generated_queries)  # Step 3 &; Step 4
        
        if not search_results or len(search_results) == 0:
            print("No relevant results found. Exiting.")
        
        else:
            #print("\nScraping content from top search results...")
            
            scraped_contents = scrape_web_content(search_results)  # Step 5
            
            if not scraped_contents or len(scraped_contents) == 0:
                print("Failed to scrape any meaningful content. Exiting.")
            
            else:
                print("\nGenerating summary...")
                
                final_summary = summarize_with_qwen(scraped_contents)  # Step 6
                
                #print("\nSummary of Results:")
                print(final_summary)
        ext_info = final_summary
        results= fe_main(process_train_df, response_train, ext_info, response=target_variable, apply_standardization=True)
        print("########################### results ################################### \n")
        print(results["fe_summary"])
        results["df_new"].to_csv("tit_test_final.csv", index=False)
        dataset_path = 'tit_test_final.csv'
        model_selection_df = pd.read_csv(dataset_path)
        # Example preprocessed data (replace with your actual data)
        X_train = model_selection_df = pd.read_csv(dataset_path).iloc[:, :-1].values  # All columns except the last one as features
        y_train = model_selection_df = pd.read_csv(dataset_path).iloc[:, -1].values   # The last column as the target

        # Step 1: Extract text from the PDF guide
        pdf_path = "Model_Selection_Guide.pdf"  # Replace with the path to your PDF
        pdf_context = extract_text_from_pdf(pdf_path)
        if not pdf_context:
            print("❌ Failed to extract text from the PDF guide.")
            return

        # Step 2: Identify the best model using Qwen and the PDF guide
        best_model_name = identify_best_model(X_train, y_train, pdf_context)
        print(f"🏆 Best Model Identified: {best_model_name}")

        # Step 3: Generate a script for the best model
        best_model_script = generate_best_model_script(best_model_name, X_train, y_train, pdf_context)
        if best_model_script:
            # Save the best model script to a JSON file
            save_script_to_json(best_model_script, "best_model_script.json")
        else:
            print("❌ Failed to generate the best model script.")

             
    else:
        print("Train and test CSV files not found in the dataset directory.")

if __name__ == "__main__":
    run_file("DataExtraction.py")
    run_file("GettingMetadata.py")
    llm_output()
