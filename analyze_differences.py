import pandas as pd
import os

def clean_company_ids(df, column_name='investee_company_beid'):
    """Clean and standardize company IDs by removing NaN and converting to string."""
    if column_name not in df.columns:
        return set()
    
    # Remove NaN values and convert to integer, then to string
    clean_ids = df[column_name].dropna()
    if clean_ids.dtype == 'float64':
        clean_ids = clean_ids.astype(int)
    return set(clean_ids.astype(str))

def analyze_differences():
    """
    Analyze differences between base Excel file and the three full CSV files.
    Identifies missing and duplicated rows and saves problematic rows to separate CSV files.
    """
    print("Starting analysis of differences between base Excel and CSV files...")
    
    # Read the base Excel file
    base_df = pd.read_excel('/Users/wbik/Downloads/label-task/250425aistartup_sdc_crunch_des_fullsample.xls')
    print(f"\nBase Excel file:")
    print(f"  Shape: {base_df.shape}")
    print(f"  Unique company IDs: {base_df['investee_company_beid'].nunique()}")
    
    # Get the set of all company IDs from the base file
    base_company_ids = clean_company_ids(base_df)
    
    # Directory for output files
    outputs_dir = '/Users/wbik/Downloads/label-task/outputs'
    
    # Files to analyze
    files_to_analyze = [
        ('full_founder.csv', 'founder'),
        ('full_executive.csv', 'executive'), 
        ('full_ipo_ma.csv', 'ipo_ma'),
        ('full_product.csv', 'product'),
        ('full_partner.csv', 'partner'),
        ('full_technology.csv', 'technology')
    ]
    
    for file_name, file_type in files_to_analyze:
        print(f"\n{'='*50}")
        print(f"Analyzing {file_name}")
        print(f"{'='*50}")
        
        file_path = os.path.join(outputs_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                # Read the CSV file with error handling
                if file_name == 'full_ipo_ma.csv':
                    # Try different parsing options for problematic file
                    try:
                        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
                    except:
                        df = pd.read_csv(file_path, sep=',', quotechar='"', on_bad_lines='skip', low_memory=False)
                elif file_name == 'full_executive.csv':
                    df = pd.read_csv(file_path, low_memory=False)
                else:
                    df = pd.read_csv(file_path)
                
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
                
                # Check if investee_company_beid column exists
                if 'investee_company_beid' not in df.columns:
                    print(f"  ERROR: 'investee_company_beid' column not found in {file_name}")
                    print(f"  Available columns: {list(df.columns)}")
                    continue
                
                # Clean company IDs for comparison
                csv_company_ids = clean_company_ids(df)
                
                print(f"  Unique company IDs: {len(csv_company_ids)}")
                
                # Find missing company IDs (in base but not in CSV)
                missing_ids = base_company_ids - csv_company_ids
                print(f"  Missing company IDs: {len(missing_ids)}")
                
                # Find extra company IDs (in CSV but not in base)
                extra_ids = csv_company_ids - base_company_ids
                print(f"  Extra company IDs: {len(extra_ids)}")
                
                # Find duplicated rows based on company ID (using cleaned IDs)
                df_clean = df.dropna(subset=['investee_company_beid']).copy()
                if df_clean['investee_company_beid'].dtype == 'float64':
                    df_clean['investee_company_beid'] = df_clean['investee_company_beid'].astype(int)
                df_clean['investee_company_beid'] = df_clean['investee_company_beid'].astype(str)
                
                duplicated_mask = df_clean.duplicated(subset=['investee_company_beid'], keep=False)
                duplicated_rows = df_clean[duplicated_mask]
                print(f"  Duplicated rows: {len(duplicated_rows)}")
                
                # Get the base data for missing IDs
                if missing_ids:
                    missing_base_data = base_df[base_df['investee_company_beid'].astype(str).isin(missing_ids)]
                    missing_file = os.path.join(outputs_dir, f'{file_type}_missing_rows.csv')
                    missing_base_data.to_csv(missing_file, index=False)
                    print(f"  Saved missing rows to: {missing_file}")
                    
                    # Show some examples of missing IDs
                    print(f"  Sample missing IDs: {list(missing_ids)[:5]}")
                
                # Save duplicated rows
                if len(duplicated_rows) > 0:
                    duplicated_file = os.path.join(outputs_dir, f'{file_type}_duplicated_rows.csv')
                    duplicated_rows.to_csv(duplicated_file, index=False)
                    print(f"  Saved duplicated rows to: {duplicated_file}")
                    
                    # Show duplicated company IDs
                    dup_company_ids = duplicated_rows['investee_company_beid'].unique()
                    print(f"  Company IDs with duplicates: {len(dup_company_ids)}")
                    print(f"  Sample duplicated IDs: {list(dup_company_ids)[:5]}")
                
                # Save extra rows (if any)
                if extra_ids:
                    extra_rows = df_clean[df_clean['investee_company_beid'].isin(extra_ids)]
                    extra_file = os.path.join(outputs_dir, f'{file_type}_extra_rows.csv')
                    extra_rows.to_csv(extra_file, index=False)
                    print(f"  Saved extra rows to: {extra_file}")
                    print(f"  Sample extra IDs: {list(extra_ids)[:5]}")
                    
            except Exception as e:
                print(f"  ERROR reading {file_name}: {e}")
                continue
                
        else:
            print(f"  ERROR: File {file_path} not found!")
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Base file has {len(base_company_ids)} unique company IDs")
    
    # Create a comprehensive summary
    summary_data = []
    for file_name, file_type in files_to_analyze:
        file_path = os.path.join(outputs_dir, file_name)
        if os.path.exists(file_path):
            try:
                if file_name == 'full_ipo_ma.csv':
                    df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
                elif file_name == 'full_executive.csv':
                    df = pd.read_csv(file_path, low_memory=False)
                else:
                    df = pd.read_csv(file_path)
                
                if 'investee_company_beid' in df.columns:
                    csv_company_ids = clean_company_ids(df)
                    missing_count = len(base_company_ids - csv_company_ids)
                    extra_count = len(csv_company_ids - base_company_ids)
                    
                    # Calculate duplicates using cleaned data
                    df_clean = df.dropna(subset=['investee_company_beid']).copy()
                    if df_clean['investee_company_beid'].dtype == 'float64':
                        df_clean['investee_company_beid'] = df_clean['investee_company_beid'].astype(int)
                    df_clean['investee_company_beid'] = df_clean['investee_company_beid'].astype(str)
                    duplicated_count = len(df_clean[df_clean.duplicated(subset=['investee_company_beid'], keep=False)])
                    
                    summary_data.append({
                        'File': file_name,
                        'Total_Rows': len(df),
                        'Unique_Company_IDs': len(csv_company_ids),
                        'Missing_IDs': missing_count,
                        'Extra_IDs': extra_count,
                        'Duplicated_Rows': duplicated_count
                    })
                else:
                    summary_data.append({
                        'File': file_name,
                        'Total_Rows': len(df),
                        'Unique_Company_IDs': 'N/A - no beid column',
                        'Missing_IDs': 'N/A',
                        'Extra_IDs': 'N/A',
                        'Duplicated_Rows': 'N/A'
                    })
            except Exception as e:
                summary_data.append({
                    'File': file_name,
                    'Total_Rows': f'ERROR: {e}',
                    'Unique_Company_IDs': 'N/A',
                    'Missing_IDs': 'N/A',
                    'Extra_IDs': 'N/A',
                    'Duplicated_Rows': 'N/A'
                })
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(outputs_dir, 'data_quality_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to: {summary_file}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    analyze_differences() 