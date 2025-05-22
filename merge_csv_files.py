import pandas as pd
import os
import numpy as np

def main():
    # Define the path to the outputs directory
    outputs_dir = '/Users/wbik/Downloads/label-task/outputs'
    
    # Define the CSV files to merge
    files_to_merge = ['executive.csv', 'founders.csv', 'product.csv', 'technology.csv']
    
    # Initialize a dictionary to store dataframes
    dataframes = {}
    
    # Read each CSV file
    for file in files_to_merge:
        file_path = os.path.join(outputs_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Extract filename without extension to use as a prefix for specific columns
            file_prefix = os.path.splitext(file)[0]
            dataframes[file_prefix] = df
            print(f"Loaded {file_path} with {len(df)} rows and {len(df.columns)} columns")
        else:
            print(f"File {file_path} not found")
    
    # Check if we have loaded any dataframes
    if not dataframes:
        print("No dataframes loaded. Exiting.")
        return
    
    # Start with the first dataframe as our base
    base_df_name = list(dataframes.keys())[0]
    merged_df = dataframes[base_df_name].copy()
    
    # Columns that are common to all dataframes and should be used for merging
    merge_keys = ['investee_company_beid', 'investee_company_name', 'investee_company_long_business_d']
    
    # Columns to sum across dataframes
    columns_to_sum = ['completion_tokens', 'prompt_tokens']
    
    # Merge the rest of the dataframes
    for df_name, df in list(dataframes.items())[1:]:
        # For each dataframe, we need to:
        # 1. Merge on the common keys
        # 2. Keep all distinct columns
        # 3. Track columns that need to be summed
        
        # Create a list of columns to sum for this merge
        sum_cols = [col for col in columns_to_sum if col in df.columns and col in merged_df.columns]
        
        # Get all columns from the dataframe we're merging in
        all_cols = df.columns.tolist()
        
        # Remove merge keys from the list to avoid duplication
        cols_to_use = [col for col in all_cols if col not in merge_keys]
        
        # Merge the dataframes on the common keys
        merged_df = pd.merge(
            merged_df, 
            df[merge_keys + cols_to_use], 
            on=merge_keys, 
            how='outer',
            suffixes=('', f'_{df_name}')
        )
    
    # Sum the token columns and create totals
    for col in columns_to_sum:
        # Find all columns that match the pattern (e.g., completion_tokens, completion_tokens_founders, etc.)
        cols_to_sum = [c for c in merged_df.columns if c == col or c.startswith(f"{col}_")]
        
        # Sum these columns into a new column
        if len(cols_to_sum) > 0:
            merged_df[f'total_{col}'] = merged_df[cols_to_sum].sum(axis=1, skipna=True)
            
            # Keep the original token columns for reference
    
    # Save the merged dataframe to a new CSV file
    output_file = os.path.join(outputs_dir, 'merged_data.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")
    print(f"Final dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    
    # Add columns from expanded_clean CSV files
    add_expanded_clean_columns(outputs_dir)

def add_expanded_clean_columns(outputs_dir):
    """
    Add columns from expanded_clean CSV files to the merged_data.csv file.
    
    New columns:
    - CS Degree: 1 if any founder has CS degree, 2 if all founders have CS degree=2, otherwise 0
    - Top10 University: 1 if any founder has a known university (not Unknown or Others), else 0
    - Prior Success IPO: 1 if any founder has prior_success_ipo=1, else 0
    - Prior Success MA: 1 if any founder has prior_success_ma=1, else 0
    - NumberofExecutives: Count of executive_name per beid
    """
    print("\nAdding columns from expanded_clean CSV files...")
    
    # Read the merged data
    merged_file = os.path.join(outputs_dir, 'merged_data.csv')
    merged_df = pd.read_csv(merged_file)
    
    # Process founders data
    founders_file = os.path.join(outputs_dir, 'founders_expanded_clean.csv')
    if os.path.exists(founders_file):
        founders_df = pd.read_csv(founders_file)
        print(f"Loaded {founders_file} with {len(founders_df)} rows")
        
        # Calculate aggregations by beid
        founders_agg = founders_df.groupby('investee_company_beid').agg(
            # CS Degree logic: 
            # - If any founder has cs_degree=1, result is 1
            # - If all founders have cs_degree=2, result is 2
            # - Otherwise, result is 0
            cs_degree_all_2=('cs_degree', lambda x: (x == 2).all()),
            cs_degree_any_1=('cs_degree', lambda x: (x == 1).any()),
            # Top10 University: 1 if any founder has graduated_university not "Unknown" or "Others"
            has_known_university=('graduated_university', 
                                lambda x: ((x != "Unknown") & (x != "Others")).any()),
            # Prior Success IPO: 1 if any founder has prior_success_ipo=1
            any_prior_ipo=('prior_success_ipo', lambda x: (x == 1).any()),
            # Prior Success MA: 1 if any founder has prior_success_ma=1
            any_prior_ma=('prior_success_ma', lambda x: (x == 1).any()),
            # Count of founders per company
            founder_count=('founder_name', 'count')
        ).reset_index()
        
        # Create CS Degree column based on the specified logic
        founders_agg['CS Degree'] = 0
        founders_agg.loc[founders_agg['cs_degree_any_1'], 'CS Degree'] = 1
        founders_agg.loc[~founders_agg['cs_degree_any_1'] & founders_agg['cs_degree_all_2'], 'CS Degree'] = 2
        
        # Create the other columns with simpler logic
        founders_agg['Top10 University'] = founders_agg['has_known_university'].astype(int)
        founders_agg['Prior Success IPO'] = founders_agg['any_prior_ipo'].astype(int)
        founders_agg['Prior Success MA'] = founders_agg['any_prior_ma'].astype(int)
        
        # Keep only the columns we need
        founders_columns = ['investee_company_beid', 'CS Degree', 'Top10 University', 
                           'Prior Success IPO', 'Prior Success MA']
        founders_agg = founders_agg[founders_columns]
        
        # Merge with the main dataframe
        merged_df = pd.merge(
            merged_df,
            founders_agg,
            on='investee_company_beid',
            how='left'
        )
        
        # Fill NA values with 0
        for col in ['CS Degree', 'Top10 University', 'Prior Success IPO', 'Prior Success MA']:
            merged_df[col] = merged_df[col].fillna(0).astype(int)
            
        print(f"Added founder columns: CS Degree, Top10 University, Prior Success IPO, Prior Success MA")
    else:
        print(f"File {founders_file} not found")
    
    # Process executives data
    executives_file = os.path.join(outputs_dir, 'executive_expanded_clean.csv')
    if os.path.exists(executives_file):
        executives_df = pd.read_csv(executives_file)
        print(f"Loaded {executives_file} with {len(executives_df)} rows")
        
        # Count executives by beid, ignoring null or empty values
        executives_df['executive_name'] = executives_df['executive_name'].replace('', np.nan)
        executives_agg = executives_df.groupby('investee_company_beid').agg(
            NumberofExecutives=('executive_name', lambda x: x.count())
        ).reset_index()
        
        # Merge with the main dataframe
        merged_df = pd.merge(
            merged_df,
            executives_agg,
            on='investee_company_beid',
            how='left'
        )
        
        # Fill NA values with 0
        merged_df['NumberofExecutives'] = merged_df['NumberofExecutives'].fillna(0).astype(int)
        
        print(f"Added executive column: NumberofExecutives")
    else:
        print(f"File {executives_file} not found")
    
    # Save the updated merged dataframe
    merged_df.to_csv(merged_file, index=False)
    print(f"Updated merged data saved to {merged_file}")
    print(f"Final dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")

if __name__ == "__main__":
    main() 