import pandas as pd
import os
import numpy as np

def main():
    # Define the path to the outputs directory
    outputs_dir = '/Users/wbik/Downloads/label-task/outputs'
    
    # Define the CSV files to merge
    files_to_merge = ['executive.csv', 'founders.csv', 'product.csv', 'technology.csv', 'partner.csv']
    
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
    
    New columns from founders_expanded_clean:
    - CS Degree: For each beid, if any row has cs_degree=1, then 1; if all rows have cs_degree=2, then 2; otherwise 0
    - Top10 University: For each beid, if any row has graduated_university not "Unknown" or "Others", then 1; else 0
    - Prior Success IPO: For each beid, if any row has prior_success_ipo=1, then 1; else 0
    - Prior Success MA: For each beid, if any row has prior_success_ma=1, then 1; else 0
    
    New columns from executive_expanded_clean:
    - NumberofExecutives: For each beid, count the number of non-null executive_name values
    
    New columns from product_expanded_clean:
    - AIProduct: For each beid, if any row has is_ai_product=1, then 1; else 0
    
    New columns from technology_expanded_clean:
    - AITech: For each beid, if any row has is_ai_tech=1, then 1; else 0
    
    New columns from partner_expanded_clean:
    - Partner: For each beid, if any row has both partner_name and collaboration_type not "Unknown", then 1; else 0
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
            # Check if all cs_degree values are 2
            all_cs_degree_2=('cs_degree', lambda x: (x == 2).all()),
            # Check if any cs_degree value is 1
            any_cs_degree_1=('cs_degree', lambda x: (x == 1).any()),
            # Check if any graduated_university is not "Unknown" or "Others"
            any_top_university=('graduated_university', lambda x: ((x != "Unknown") & (x != "Others")).any()),
            # Check if any prior_success_ipo is 1
            any_prior_ipo=('prior_success_ipo', lambda x: (x == 1).any()),
            # Check if any prior_success_ma is 1
            any_prior_ma=('prior_success_ma', lambda x: (x == 1).any())
        ).reset_index()
        
        # Create CS Degree column based on the specified logic
        # If any cs_degree is 1, set to 1
        # If all cs_degree are 2, set to 2
        # Otherwise, set to 0
        founders_agg['CS Degree'] = 0
        # First check for any cs_degree = 1 (this takes precedence)
        founders_agg.loc[founders_agg['any_cs_degree_1'], 'CS Degree'] = 1
        # Then, only for rows that don't have any cs_degree = 1 but all = 2
        founders_agg.loc[(~founders_agg['any_cs_degree_1']) & founders_agg['all_cs_degree_2'], 'CS Degree'] = 2
        
        # Create other founder-related columns
        founders_agg['Top10 University'] = founders_agg['any_top_university'].astype(int)
        founders_agg['Prior Success IPO'] = founders_agg['any_prior_ipo'].astype(int)
        founders_agg['Prior Success MA'] = founders_agg['any_prior_ma'].astype(int)
        
        # Select only the columns we need for merging
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
        
        # Replace empty strings with NaN for proper counting
        executives_df['executive_name'] = executives_df['executive_name'].replace('', np.nan)
        
        # Count executives by beid, ignoring null values
        executives_agg = executives_df.groupby('investee_company_beid').agg(
            NumberofExecutives=('executive_name', lambda x: x.notna().sum())
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
    
    # Process product data
    product_file = os.path.join(outputs_dir, 'product_expanded_clean.csv')
    if os.path.exists(product_file):
        product_df = pd.read_csv(product_file)
        print(f"Loaded {product_file} with {len(product_df)} rows")
        
        # Check if any product has is_ai_product=1 for each beid
        product_agg = product_df.groupby('investee_company_beid').agg(
            any_ai_product=('is_ai_product', lambda x: (x == 1).any())
        ).reset_index()
        
        # Convert boolean to integer
        product_agg['AIProduct'] = product_agg['any_ai_product'].astype(int)
        
        # Select only the columns we need for merging
        product_columns = ['investee_company_beid', 'AIProduct']
        product_agg = product_agg[product_columns]
        
        # Merge with the main dataframe
        merged_df = pd.merge(
            merged_df,
            product_agg,
            on='investee_company_beid',
            how='left'
        )
        
        # Fill NA values with 0
        merged_df['AIProduct'] = merged_df['AIProduct'].fillna(0).astype(int)
        
        print(f"Added product column: AIProduct")
    else:
        print(f"File {product_file} not found")
    
    # Process technology data
    tech_file = os.path.join(outputs_dir, 'technology_expanded_clean.csv')
    if os.path.exists(tech_file):
        tech_df = pd.read_csv(tech_file)
        print(f"Loaded {tech_file} with {len(tech_df)} rows")
        
        # Check if any technology has is_ai_tech=1 for each beid
        tech_agg = tech_df.groupby('investee_company_beid').agg(
            any_ai_tech=('is_ai_tech', lambda x: (x == 1).any())
        ).reset_index()
        
        # Convert boolean to integer
        tech_agg['AITech'] = tech_agg['any_ai_tech'].astype(int)
        
        # Select only the columns we need for merging
        tech_columns = ['investee_company_beid', 'AITech']
        tech_agg = tech_agg[tech_columns]
        
        # Merge with the main dataframe
        merged_df = pd.merge(
            merged_df,
            tech_agg,
            on='investee_company_beid',
            how='left'
        )
        
        # Fill NA values with 0
        merged_df['AITech'] = merged_df['AITech'].fillna(0).astype(int)
        
        print(f"Added technology column: AITech")
    else:
        print(f"File {tech_file} not found")
    
    # Process partner data
    partner_file = os.path.join(outputs_dir, 'partner_expanded_clean.csv')
    if os.path.exists(partner_file):
        partner_df = pd.read_csv(partner_file)
        print(f"Loaded {partner_file} with {len(partner_df)} rows")
        
        # Check if any row has both partner_name and collaboration_type not "Unknown"
        partner_df['has_valid_partner'] = (partner_df['partner_name'] != "Unknown") & (partner_df['collaboration_type'] != "Unknown")
        
        # Aggregate by beid
        partner_agg = partner_df.groupby('investee_company_beid').agg(
            has_partner=('has_valid_partner', lambda x: x.any())
        ).reset_index()
        
        # Convert boolean to integer
        partner_agg['Partner'] = partner_agg['has_partner'].astype(int)
        
        # Select only the columns we need for merging
        partner_columns = ['investee_company_beid', 'Partner']
        partner_agg = partner_agg[partner_columns]
        
        # Merge with the main dataframe
        merged_df = pd.merge(
            merged_df,
            partner_agg,
            on='investee_company_beid',
            how='left'
        )
        
        # Fill NA values with 0
        merged_df['Partner'] = merged_df['Partner'].fillna(0).astype(int)
        
        print(f"Added partner column: Partner")
    else:
        print(f"File {partner_file} not found")
    
    # Add Genuine column that equals 1 if either AIProduct or Partner equals 1, and 0 otherwise
    merged_df['Genuine'] = ((merged_df['AIProduct'] == 1) | (merged_df['Partner'] == 1)).astype(int)
    print(f"Added Genuine column: 1 if AIProduct=1 or Partner=1, 0 otherwise")
    
    # Save the updated merged dataframe
    merged_df.to_csv(merged_file, index=False)
    print(f"Updated merged data saved to {merged_file}")
    print(f"Final dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")

if __name__ == "__main__":
    main() 