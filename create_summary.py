import pandas as pd
import os
import io

def create_dataset_summary(data_folder='data', output_file='ddataset_summary.txt'):
    """
    Analyzes all CSV and Parquet files in a specified folder and compiles the results
    into a single summary text file.

    For each file, it extracts:
    1. Schema (column names, non-null counts, dtypes).
    2. Head Sample (first 5 rows).
    3. Random Sample (5 random rows).
    4. Descriptive Statistics for numerical and categorical columns.
    5. Value Counts for categorical columns.

    Args:
        data_folder (str): The path to the folder containing the data files.
        output_file (str): The name of the file to save the summary to.
    """
    # --- Helper function to create separators ---
    def write_separator(f, char='=', length=80, title=None):
        if title:
            f.write(f"\n{char * length}\n")
            f.write(f"{title.center(length)}\n")
            f.write(f"{char * length}\n\n")
        else:
            f.write(f"\n{'-' * length}\n\n")

    # --- Check if the data directory exists ---
    if not os.path.isdir(data_folder):
        print(f"Error: The directory '{data_folder}' was not found.")
        print("Please create a 'data' folder and place your CSV/Parquet files inside it.")
        return

    # --- Get list of files to process ---
    files_to_process = [
        f for f in os.listdir(data_folder)
        if f.endswith('.csv') or f.endswith('.parquet')
    ]

    if not files_to_process:
        print(f"No CSV or Parquet files found in the '{data_folder}' directory.")
        return

    print(f"Found {len(files_to_process)} files to process. Generating summary...")

    # --- Open the output file and start processing ---
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DATASET SUMMARY REPORT\n")
        f.write(f"Generated for files in the '{data_folder}' folder.\n")

        for filename in files_to_process:
            file_path = os.path.join(data_folder, filename)
            write_separator(f, char='=', title=f"FILE: {filename}")

            try:
                # --- Read the file based on its extension ---
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else: # .parquet
                    df = pd.read_parquet(file_path)

                f.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

                # --- 1. Schema / Info ---
                write_separator(f, title="1. Schema / Data Types")
                # Redirect pandas's df.info() output to a string buffer
                buffer = io.StringIO()
                df.info(buf=buffer)
                f.write(buffer.getvalue())
                f.write("\n")


                # --- 2. Head Sample ---
                write_separator(f, title="2. Head Sample (First 5 Rows)")
                f.write(df.head().to_string())
                f.write("\n")

                # --- 3. Random Sample ---
                write_separator(f, title="3. Random Sample (5 Rows)")
                # Ensure sample size is not larger than the number of rows
                sample_size = min(5, len(df))
                if sample_size > 0:
                    f.write(df.sample(sample_size).to_string())
                else:
                    f.write("Dataset is empty, no random sample to show.")
                f.write("\n")

                # --- 4. Descriptive Statistics ---
                write_separator(f, title="4. Descriptive Statistics")
                # Use include='all' to get stats for both numeric and object dtypes
                f.write(df.describe(include='all').to_string())
                f.write("\n")

                # --- 5. Value Counts for Categorical Columns ---
                # write_separator(f, title="5. Value Counts for Categorical Columns")
                # # Select columns with 'object' or 'category' dtype
                # categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                # if not categorical_cols.empty:
                #     for col in categorical_cols:
                #         f.write(f"\n--- Column: {col} ---\n")
                #         f.write(df[col].value_counts(dropna=False).to_string())
                #         f.write("\n")
                # else:
                #     f.write("No categorical columns found in this dataset.\n")


            except Exception as e:
                f.write(f"\n!!! ERROR: Could not process file '{filename}'. !!!\n")
                f.write(f"Reason: {e}\n")

    print(f"\nSummary successfully generated and saved to '{output_file}'")

if __name__ == '__main__':
    # To run this script:
    # 1. Save it as a Python file (e.g., `create_summary.py`).
    # 2. Create a folder named 'data' in the same directory.
    # 3. Place your .csv and .parquet files into the 'data' folder.
    # 4. Run the script from your terminal: python create_summary.py
    create_dataset_summary()
