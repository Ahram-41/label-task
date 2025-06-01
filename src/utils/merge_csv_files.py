import pandas as pd
import os
import numpy as np
import re

def truncate_tool_calls_columns(df, max_length=500):
    """
    Truncate tool_calls columns to prevent CSV formatting issues.
    
    Args:
        df: DataFrame to process
        max_length: Maximum length for tool_calls content (default 500 characters)
    
    Returns:
        DataFrame with truncated tool_calls columns
    """
    tool_calls_columns = [col for col in df.columns if 'tool_calls' in col.lower()]
    
    for col in tool_calls_columns:
        if col in df.columns:
            # Convert to string and truncate
            df[col] = df[col].astype(str)
            # Truncate long entries
            df[col] = df[col].apply(lambda x: x[:max_length] + '...' if len(str(x)) > max_length else x)
            # Replace problematic characters that might break CSV
            df[col] = df[col].str.replace('\n', ' ').str.replace('\r', ' ').str.replace('"', "'")
            print(f"Truncated and cleaned column: {col}")
    
    return df

def clean_source_data(df, file_name):
    """
    Clean source data by fixing problematic rows instead of removing them.
    
    Args:
        df: DataFrame to clean
        file_name: Name of the file for logging
    
    Returns:
        Cleaned DataFrame
    """
    original_count = len(df)
    df_clean = df.copy()
    
    # Track what we're fixing
    fixes_applied = []
    
    # Fix missing token columns by filling with 0
    if 'completion_tokens' in df_clean.columns:
        completion_na_count = df_clean['completion_tokens'].isna().sum()
        prompt_na_count = df_clean['prompt_tokens'].isna().sum()
        
        if completion_na_count > 0:
            df_clean['completion_tokens'] = df_clean['completion_tokens'].fillna(0)
            fixes_applied.append(f"filled {completion_na_count} missing completion_tokens with 0")
        
        if prompt_na_count > 0:
            df_clean['prompt_tokens'] = df_clean['prompt_tokens'].fillna(0)
            fixes_applied.append(f"filled {prompt_na_count} missing prompt_tokens with 0")
    
    # Fix missing company names by using beid as fallback
    company_name_na_count = df_clean['investee_company_name'].isna().sum()
    if company_name_na_count > 0:
        # Use beid as company name when company name is missing
        mask = df_clean['investee_company_name'].isna()
        df_clean.loc[mask, 'investee_company_name'] = 'Company_' + df_clean.loc[mask, 'investee_company_beid'].astype(str)
        fixes_applied.append(f"filled {company_name_na_count} missing company names with beid-based names")
    
    # For content columns, fill with appropriate defaults
    if 'executives' in df_clean.columns:
        exec_na_count = df_clean['executives'].isna().sum()
        if exec_na_count > 0:
            df_clean['executives'] = df_clean['executives'].fillna('No data available')
            fixes_applied.append(f"filled {exec_na_count} missing executives with default text")
    
    if 'founders' in df_clean.columns:
        founder_na_count = df_clean['founders'].isna().sum()
        if founder_na_count > 0:
            df_clean['founders'] = df_clean['founders'].fillna('No data available')
            fixes_applied.append(f"filled {founder_na_count} missing founders with default text")
    
    if 'products' in df_clean.columns:
        product_na_count = df_clean['products'].isna().sum()
        if product_na_count > 0:
            df_clean['products'] = df_clean['products'].fillna('No data available')
            fixes_applied.append(f"filled {product_na_count} missing products with default text")
    
    if 'technology' in df_clean.columns:
        tech_na_count = df_clean['technology'].isna().sum()
        if tech_na_count > 0:
            df_clean['technology'] = df_clean['technology'].fillna('No data available')
            fixes_applied.append(f"filled {tech_na_count} missing technology with default text")
    
    if 'partners' in df_clean.columns:
        partner_na_count = df_clean['partners'].isna().sum()
        if partner_na_count > 0:
            df_clean['partners'] = df_clean['partners'].fillna('No data available')
            fixes_applied.append(f"filled {partner_na_count} missing partners with default text")
    
    # Only remove rows that are completely unusable (missing beid)
    unusable_rows = df_clean['investee_company_beid'].isna()
    unusable_count = unusable_rows.sum()
    
    if unusable_count > 0:
        df_clean = df_clean[~unusable_rows]
        fixes_applied.append(f"removed {unusable_count} rows with missing beid (unusable)")
    
    # Report what was fixed
    if fixes_applied:
        print(f"Fixed {file_name}: {'; '.join(fixes_applied)} ({original_count} -> {len(df_clean)})")
    else:
        print(f"No fixes needed for {file_name}: {len(df_clean)} rows")
    
    return df_clean

def main():
    # Define the path to the outputs directory
    outputs_dir = '/Users/wbik/Downloads/label-task'
    
    # Define the CSV files to merge
    # files_to_merge = ['executive.csv', 'founder.csv', 'product.csv', 'technology.csv', 'partner.csv']
    files_to_merge = ['full_executive.csv', 'full_founder.csv', 'full_product.csv', 'full_technology.csv', 'full_partner.csv']
    
    # Initialize a dictionary to store dataframes
    dataframes = {}
    
    # Read each CSV file
    for file in files_to_merge:
        file_path = os.path.join(outputs_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Clean the data before processing
            df = clean_source_data(df, file)
            # Extract filename without extension to use as a prefix for specific columns
            file_prefix = os.path.splitext(file)[0]
            dataframes[file_prefix] = df
            print(f"Loaded and cleaned {file_path} with {len(df)} rows and {len(df.columns)} columns")
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
    merge_keys = ['investee_company_beid', 'investee_company_name']
    
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
            how='left',
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
    
    # Merge IPO/M&A data
    ipo_ma_file = os.path.join(outputs_dir, 'full_ipo_ma.csv')
    if os.path.exists(ipo_ma_file):
        print(f"Reading IPO/M&A data from {ipo_ma_file}")
        ipo_ma_df = pd.read_csv(ipo_ma_file)
        
        # Clean the IPO/M&A data
        ipo_ma_df = clean_source_data(ipo_ma_df, 'ipo_ma')
        
        # Extract IPO/M&A info from founders column
        def extract_value(row, key):
            try:
                # Find the key-value pair in the founders string
                pattern = f"{key}=(\\d|'')"
                match = re.search(pattern, row)
                if match:
                    value = match.group(1)
                    return 1 if value == '1' else 0
            except:
                pass
            return 0
            
        def extract_date(row, key):
            try:
                # Find the date value in the founders string
                pattern = f"{key}_date='([^']*)"
                match = re.search(pattern, row)
                if match:
                    return match.group(1)
            except:
                pass
            return ''
            
        # Extract values from founders column
        ipo_ma_df['ipo'] = ipo_ma_df['founders'].apply(lambda x: extract_value(x, 'ipo'))
        ipo_ma_df['ma'] = ipo_ma_df['founders'].apply(lambda x: extract_value(x, 'ma'))
        ipo_ma_df['ipo_date'] = ipo_ma_df['founders'].apply(lambda x: extract_date(x, 'ipo'))
        ipo_ma_df['ma_date'] = ipo_ma_df['founders'].apply(lambda x: extract_date(x, 'ma'))
        
        # Select only needed columns for merging
        ipo_ma_columns = ['investee_company_beid', 'ipo', 'ipo_date', 'ma', 'ma_date', 'tool_calls', 'completion_tokens', 'prompt_tokens']
        ipo_ma_df = ipo_ma_df[ipo_ma_columns]
        
        # Rename token columns to add suffix for distinction
        ipo_ma_df = ipo_ma_df.rename(columns={
            'tool_calls': 'tool_calls_ipo_ma',
            'completion_tokens': 'completion_tokens_ipo_ma',
            'prompt_tokens': 'prompt_tokens_ipo_ma'
        })
        
        # Merge with main dataframe
        merged_df = pd.merge(
            merged_df,
            ipo_ma_df,
            on='investee_company_beid',
            how='left'
        )
        
        # Fill NA values with 0 for boolean columns and empty string for dates
        merged_df['ipo'] = merged_df['ipo'].fillna(0).astype(int)
        merged_df['ma'] = merged_df['ma'].fillna(0).astype(int)
        merged_df['ipo_date'] = merged_df['ipo_date'].fillna('')
        merged_df['ma_date'] = merged_df['ma_date'].fillna('')
        
        # Fill NA values for token columns with 0
        merged_df['completion_tokens_ipo_ma'] = merged_df['completion_tokens_ipo_ma'].fillna(0).astype(int)
        merged_df['prompt_tokens_ipo_ma'] = merged_df['prompt_tokens_ipo_ma'].fillna(0).astype(int)
        merged_df['tool_calls_ipo_ma'] = merged_df['tool_calls_ipo_ma'].fillna('')
        
        print(f"Added IPO/M&A columns: ipo, ipo_date, ma, ma_date, tool_calls_ipo_ma, completion_tokens_ipo_ma, prompt_tokens_ipo_ma")
        
        # Recalculate total token columns to include IPO/M&A tokens
        for col in columns_to_sum:
            # Find all columns that match the pattern (including the new ipo_ma columns)
            cols_to_sum = [c for c in merged_df.columns if c == col or c.startswith(f"{col}_")]
            
            # Sum these columns into the total column
            if len(cols_to_sum) > 0:
                merged_df[f'total_{col}'] = merged_df[cols_to_sum].sum(axis=1, skipna=True)
                print(f"Updated total_{col} to include IPO/M&A tokens")
    else:
        print(f"File {ipo_ma_file} not found")
    
    # Save the merged dataframe to a new CSV file
    output_file = os.path.join(outputs_dir, 'full_merged_data.csv')
    # output_file = os.path.join(outputs_dir, 'merged_data.csv')
    merged_df = truncate_tool_calls_columns(merged_df)
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
    merged_file = os.path.join(outputs_dir, 'full_merged_data.csv')
    merged_df = pd.read_csv(merged_file)
    
    # Process founders data
    founders_file = os.path.join(outputs_dir, 'full_founder_expanded_clean.csv')
    if os.path.exists(founders_file):
        founders_df = pd.read_csv(founders_file)
        print(f"Loaded {founders_file} with {len(founders_df)} rows")
        
        # Filter out rows with Unknown or blank founder names before calculating team size
        original_founder_count = len(founders_df)
        
        # Remove rows where founder_name is Unknown, blank, or null
        founders_df = founders_df[
            (founders_df['founder_name'].notna()) & 
            (founders_df['founder_name'] != '') & 
            (founders_df['founder_name'] != 'Unknown')
        ]
        
        filtered_count = len(founders_df)
        removed_count = original_founder_count - filtered_count
        
        if removed_count > 0:
            print(f"Filtered out {removed_count} founders with Unknown/blank names for TeamSize calculation ({original_founder_count} -> {filtered_count})")
        
        # Calculate aggregations by beid
        founders_agg = founders_df.groupby('investee_company_beid').agg(
            # Check if all cs_degree values are 2
            all_cs_degree_2=('cs_degree', lambda x: (x == 2).all()),
            # Check if any cs_degree value is 1
            any_cs_degree_1=('cs_degree', lambda x: (x == 1).any()),
            # Check if any graduated_university is not "Unknown" or "Others"
            any_top_university=('is_top10_university', lambda x: ((x == 1)).any()),
            # Check if any prior_success_ipo is 1
            any_prior_ipo=('prior_success_ipo', lambda x: (x == 1).any()),
            # Check if any prior_success_ma is 1
            any_prior_ma=('prior_success_ma', lambda x: (x == 1).any()),
            # Count number of founding team members (rows per company) - now only counting valid founders
            team_size=('investee_company_beid', 'count')
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
        founders_agg['TeamSize'] = founders_agg['team_size']
        
        # Select only the columns we need for merging
        founders_columns = ['investee_company_beid', 'CS Degree', 'Top10 University', 
                           'Prior Success IPO', 'Prior Success MA', 'TeamSize']
        founders_agg = founders_agg[founders_columns]
        
        # Merge with the main dataframe
        merged_df = pd.merge(
            merged_df,
            founders_agg,
            on='investee_company_beid',
            how='left'
        )
        
        # Fill NA values with 0
        for col in ['CS Degree', 'Top10 University', 'Prior Success IPO', 'Prior Success MA', 'TeamSize']:
            merged_df[col] = merged_df[col].fillna(0).astype(int)
            
        print(f"Added founder columns: CS Degree, Top10 University, Prior Success IPO, Prior Success MA, TeamSize")
    else:
        print(f"File {founders_file} not found")
    
    # Process executives data
    executives_file = os.path.join(outputs_dir, 'full_executive_expanded_clean.csv')
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
        
        # Use TeamSize as substitute when NumberofExecutives is 0
        if 'TeamSize' in merged_df.columns:
            mask = merged_df['NumberofExecutives'] == 0
            merged_df.loc[mask, 'NumberofExecutives'] = merged_df.loc[mask, 'TeamSize']
            print(f"Substituted TeamSize for NumberofExecutives where NumberofExecutives was 0")
        
        print(f"Added executive column: NumberofExecutives")
    else:
        print(f"File {executives_file} not found")
    
    # Process product data
    product_file = os.path.join(outputs_dir, 'full_product_expanded_clean.csv')
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
        # Add default AIProduct column with all 0s if file not found
        merged_df['AIProduct'] = 0
        print(f"Added default AIProduct column with all 0s")
    
    # Process technology data
    tech_file = os.path.join(outputs_dir, 'full_technology_expanded_clean.csv')
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
        # Add default AITech column with all 0s if file not found
        merged_df['AITech'] = 0
        print(f"Added default AITech column with all 0s")
    
    # Process partner data
    partner_file = os.path.join(outputs_dir, 'full_partner_expanded_clean.csv')
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
        # Add default Partner column with all 0s if file not found
        merged_df['Partner'] = 0
        print(f"Added default Partner column with all 0s")
    
    # Add Genuine column that equals 1 if either AIProduct or Partner equals 1, and 0 otherwise
    merged_df['Genuine'] = ((merged_df['AIProduct'] == 1) | (merged_df['Partner'] == 1)).astype(int)
    print(f"Added Genuine column: 1 if AIProduct=1 or Partner=1, 0 otherwise")
    
    # Save the updated merged dataframe
    merged_df = truncate_tool_calls_columns(merged_df)
    merged_df.to_csv(merged_file, index=False)
    print(f"Updated merged data saved to {merged_file}")
    print(f"Final dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")

if __name__ == "__main__":
    main() 