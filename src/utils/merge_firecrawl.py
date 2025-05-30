"""
merge all other outputs/llm_firecrawl_output_{task_name}.csv,
into outputs/llm_firecrawl_output_f_ipo_mna.csv.
ipo_mna: keep the original columns
investee_company_beid,investee_company_name,investee_company_long_business_d, ipo,ipo_date,ma,ma_date,
other csvs: save {task_name}_success_search,{task_name}_finalAnalysis,{task_name}_source,{task_name}_processing_time
and save all distinct columns of each csv to the final csv.
Important columns (founders, executives, products, technologies, partners) should not have prefixes.
"""
import pandas as pd
import os

TASK_NAME = ["f_ipo_mna", "f_founder", "f_executive", "f_product", "f_technology", "f_partner"]

def merge_firecrawl_csvs():
    """
    Merge all firecrawl CSV files into a single comprehensive CSV file.
    """
    
    # Base file path
    base_path = "outputs/llm_firecrawl_output_{}.csv"
    
    # Load the main CSV (f_ipo_mna) - this will be our base
    main_csv_path = base_path.format("f_ipo_mna")
    
    if not os.path.exists(main_csv_path):
        raise FileNotFoundError(f"Main CSV file not found: {main_csv_path}")
    
    print(f"Loading main CSV: {main_csv_path}")
    main_df = pd.read_csv(main_csv_path)
    print(f"Main CSV shape: {main_df.shape}")
    
    # Keep the original columns from f_ipo_mna
    # Based on the inspection, the key columns to keep are:
    base_columns = ['investee_company_beid', 'investee_company_name', 'investee_company_long_business_d', 
                   'ipo', 'ipo_date', 'ma', 'ma_date']
    
    # Check which base columns actually exist in the main file
    existing_base_columns = [col for col in base_columns if col in main_df.columns]
    print(f"Base columns found: {existing_base_columns}")
    
    # Start with the base dataframe, keeping all original columns for now
    merged_df = main_df.copy()
    
    # Define important columns that should not have prefixes
    important_columns = ['founders', 'executives', 'products', 'technologies', 'partners']
    
    # Process other CSV files
    for task_name in TASK_NAME:
        if task_name == "f_ipo_mna":
            continue  # Skip the main file
            
        csv_path = base_path.format(task_name)
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            continue
            
        print(f"Processing: {csv_path}")
        task_df = pd.read_csv(csv_path)
        print(f"  Shape: {task_df.shape}")
        
        # Get the task name without 'f_' prefix for column naming
        task_short = task_name.replace('f_', '')
        
        # Define the columns we want to extract and rename (with prefixes)
        columns_to_extract = {
            'success_search': f'{task_short}_success_search',
            'finalAnalysis': f'{task_short}_finalAnalysis', 
            'source': f'{task_short}_source',
            'processing_time': f'{task_short}_processing_time'
        }
        
        # Check which columns actually exist in this CSV
        existing_columns = {}
        for original_col, new_col in columns_to_extract.items():
            if original_col in task_df.columns:
                existing_columns[original_col] = new_col
            else:
                print(f"  Warning: Column '{original_col}' not found in {csv_path}")
        
        # Check for important columns that should not have prefixes
        important_cols_found = {}
        for important_col in important_columns:
            if important_col in task_df.columns:
                important_cols_found[important_col] = important_col  # Keep original name
                print(f"  Found important column: {important_col}")
        
        # Combine all columns to extract
        all_columns_to_extract = {**existing_columns, **important_cols_found}
        
        if not all_columns_to_extract:
            print(f"  No target columns found in {csv_path}")
            continue
            
        # Select and rename the columns
        task_subset = task_df[['investee_company_beid'] + list(all_columns_to_extract.keys())].copy()
        task_subset = task_subset.rename(columns=all_columns_to_extract)
        
        print(f"  Extracted columns: {list(all_columns_to_extract.values())}")
        
        # Merge with main dataframe on investee_company_beid
        merged_df = merged_df.merge(
            task_subset, 
            on='investee_company_beid', 
            how='left',
            suffixes=('', f'_{task_short}')
        )
        
        print(f"  Merged shape: {merged_df.shape}")
    
    # Also add any additional distinct columns from each CSV that aren't in the base columns
    print("\nChecking for additional distinct columns...")
    for task_name in TASK_NAME:
        csv_path = base_path.format(task_name)
        
        if not os.path.exists(csv_path):
            continue
            
        task_df = pd.read_csv(csv_path)
        task_short = task_name.replace('f_', '')
        
        # Find columns that are not in our standard list and not already processed
        standard_columns = ['investee_company_beid', 'investee_company_name', 'investee_company_long_business_d',
                          'success_search', 'finalAnalysis', 'source', 'processing_time', 'error'] + important_columns
        
        distinct_columns = [col for col in task_df.columns if col not in standard_columns]
        
        if distinct_columns:
            print(f"  Found distinct columns in {task_name}: {distinct_columns}")
            
            # Add these distinct columns with task prefix
            for col in distinct_columns:
                if col not in merged_df.columns:
                    # Select this column and merge it
                    distinct_subset = task_df[['investee_company_beid', col]].copy()
                    
                    # Add task prefix if it's from a specific task and not the main one
                    if task_name != "f_ipo_mna":
                        new_col_name = f"{task_short}_{col}"
                        distinct_subset = distinct_subset.rename(columns={col: new_col_name})
                    else:
                        new_col_name = col
                    
                    merged_df = merged_df.merge(
                        distinct_subset,
                        on='investee_company_beid',
                        how='left',
                        suffixes=('', f'_dup')
                    )
                    
                    print(f"    Added column: {new_col_name}")
    
    # Save the merged dataframe
    output_path = "outputs/merged_firecrawl_output.csv"
    merged_df.to_csv(output_path, index=False)
    
    print(f"\nMerged CSV saved to: {output_path}")
    print(f"Final shape: {merged_df.shape}")
    print(f"Final columns: {list(merged_df.columns)}")
    
    return merged_df

if __name__ == "__main__":
    merged_data = merge_firecrawl_csvs()
