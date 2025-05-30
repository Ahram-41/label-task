"""
Compare summary statistics between Tavily (merged_data.csv) and Firecrawl (merged_firecrawl_data.csv) datasets
"""
import pandas as pd
import numpy as np

def load_datasets():
    """Load both datasets"""
    try:
        tavily_df = pd.read_csv('outputs/merged_data.csv')
        print(f"Loaded Tavily dataset: {len(tavily_df)} rows × {len(tavily_df.columns)} columns")
    except FileNotFoundError:
        print("ERROR: Tavily dataset (outputs/merged_data.csv) not found")
        return None, None
    
    try:
        firecrawl_df = pd.read_csv('outputs/merged_firecrawl_data.csv')
        print(f"Loaded Firecrawl dataset: {len(firecrawl_df)} rows × {len(firecrawl_df.columns)} columns")
    except FileNotFoundError:
        print("ERROR: Firecrawl dataset (outputs/merged_firecrawl_data.csv) not found")
        return None, None
    
    return tavily_df, firecrawl_df

def analyze_column_availability(tavily_df, firecrawl_df, target_columns):
    """Check which target columns are available in each dataset"""
    print("\nCOLUMN AVAILABILITY CHECK:")
    print("=" * 50)
    
    tavily_available = []
    firecrawl_available = []
    
    for col in target_columns:
        tavily_has = col in tavily_df.columns
        firecrawl_has = col in firecrawl_df.columns
        
        if tavily_has:
            tavily_available.append(col)
        if firecrawl_has:
            firecrawl_available.append(col)
            
        status = "OK" if (tavily_has and firecrawl_has) else "MISSING"
        print(f"{status:7} {col:20} | Tavily: {'✓' if tavily_has else '✗':3} | Firecrawl: {'✓' if firecrawl_has else '✗':3}")
    
    return tavily_available, firecrawl_available

def compare_binary_columns(tavily_df, firecrawl_df, columns):
    """Compare binary columns (0/1 values)"""
    print("\nBINARY COLUMNS COMPARISON:")
    print("=" * 80)
    print(f"{'Column':<20} | {'Tavily (Yes/Total)':<20} | {'Firecrawl (Yes/Total)':<22} | {'Difference':<12}")
    print("-" * 80)
    
    results = {}
    
    for col in columns:
        tavily_available = col in tavily_df.columns
        firecrawl_available = col in firecrawl_df.columns
        
        if tavily_available and firecrawl_available:
            # Calculate statistics for both datasets
            tavily_yes = (tavily_df[col] == 1).sum()
            tavily_total = len(tavily_df)
            tavily_pct = (tavily_yes / tavily_total * 100) if tavily_total > 0 else 0
            
            firecrawl_yes = (firecrawl_df[col] == 1).sum()
            firecrawl_total = len(firecrawl_df)
            firecrawl_pct = (firecrawl_yes / firecrawl_total * 100) if firecrawl_total > 0 else 0
            
            diff = firecrawl_pct - tavily_pct
            
            print(f"{col:<20} | {tavily_yes}/{tavily_total} ({tavily_pct:5.1f}%){'':<3} | {firecrawl_yes}/{firecrawl_total} ({firecrawl_pct:5.1f}%){'':<5} | {diff:+6.1f}%")
            
            results[col] = {
                'tavily_count': tavily_yes,
                'tavily_total': tavily_total,
                'tavily_pct': tavily_pct,
                'firecrawl_count': firecrawl_yes,
                'firecrawl_total': firecrawl_total,
                'firecrawl_pct': firecrawl_pct,
                'difference': diff
            }
        elif tavily_available:
            tavily_yes = (tavily_df[col] == 1).sum()
            tavily_total = len(tavily_df)
            tavily_pct = (tavily_yes / tavily_total * 100) if tavily_total > 0 else 0
            print(f"{col:<20} | {tavily_yes}/{tavily_total} ({tavily_pct:5.1f}%){'':<3} | {'N/A':<22} | {'N/A':<12}")
        elif firecrawl_available:
            firecrawl_yes = (firecrawl_df[col] == 1).sum()
            firecrawl_total = len(firecrawl_df)
            firecrawl_pct = (firecrawl_yes / firecrawl_total * 100) if firecrawl_total > 0 else 0
            print(f"{col:<20} | {'N/A':<20} | {firecrawl_yes}/{firecrawl_total} ({firecrawl_pct:5.1f}%){'':<5} | {'N/A':<12}")
    
    return results

def compare_categorical_columns(tavily_df, firecrawl_df, columns):
    """Compare categorical columns (like CS Degree with values 0,1,2)"""
    print("\nCATEGORICAL COLUMNS COMPARISON:")
    print("=" * 60)
    
    for col in columns:
        if col in tavily_df.columns and col in firecrawl_df.columns:
            print(f"\nAnalyzing {col}:")
            
            # Get value counts for both
            tavily_counts = tavily_df[col].value_counts().sort_index()
            firecrawl_counts = firecrawl_df[col].value_counts().sort_index()
            
            # Get all unique values
            all_values = sorted(set(list(tavily_counts.index) + list(firecrawl_counts.index)))
            
            print(f"{'Value':<8} | {'Tavily':<15} | {'Firecrawl':<15} | {'Difference':<12}")
            print("-" * 55)
            
            for value in all_values:
                tavily_count = tavily_counts.get(value, 0)
                firecrawl_count = firecrawl_counts.get(value, 0)
                
                tavily_pct = (tavily_count / len(tavily_df) * 100) if len(tavily_df) > 0 else 0
                firecrawl_pct = (firecrawl_count / len(firecrawl_df) * 100) if len(firecrawl_df) > 0 else 0
                
                diff = firecrawl_pct - tavily_pct
                
                print(f"{value:<8} | {tavily_count} ({tavily_pct:4.1f}%){'':<4} | {firecrawl_count} ({firecrawl_pct:4.1f}%){'':<4} | {diff:+6.1f}%")

def compare_continuous_columns(tavily_df, firecrawl_df, columns):
    """Compare continuous columns like NumberofExecutives"""
    print("\nCONTINUOUS COLUMNS COMPARISON:")
    print("=" * 70)
    
    for col in columns:
        if col in tavily_df.columns and col in firecrawl_df.columns:
            print(f"\nAnalyzing {col}:")
            
            # Calculate statistics
            tavily_stats = {
                'mean': tavily_df[col].mean(),
                'median': tavily_df[col].median(),
                'std': tavily_df[col].std(),
                'min': tavily_df[col].min(),
                'max': tavily_df[col].max()
            }
            
            firecrawl_stats = {
                'mean': firecrawl_df[col].mean(),
                'median': firecrawl_df[col].median(),
                'std': firecrawl_df[col].std(),
                'min': firecrawl_df[col].min(),
                'max': firecrawl_df[col].max()
            }
            
            print(f"{'Metric':<10} | {'Tavily':<12} | {'Firecrawl':<12} | {'Difference':<12}")
            print("-" * 50)
            
            for metric in ['mean', 'median', 'std', 'min', 'max']:
                tavily_val = tavily_stats[metric]
                firecrawl_val = firecrawl_stats[metric]
                diff = firecrawl_val - tavily_val if not pd.isna(tavily_val) and not pd.isna(firecrawl_val) else np.nan
                
                print(f"{metric:<10} | {tavily_val:<12.2f} | {firecrawl_val:<12.2f} | {diff:+12.2f}")

def generate_summary_insights(tavily_df, firecrawl_df):
    """Generate key insights from the comparison"""
    print("\nKEY INSIGHTS:")
    print("=" * 50)
    
    # Dataset sizes
    print(f"Dataset Sizes:")
    print(f"   • Tavily: {len(tavily_df)} companies")
    print(f"   • Firecrawl: {len(firecrawl_df)} companies")
    print(f"   • Size difference: {len(firecrawl_df) - len(tavily_df):+d} companies")
    
    # Common companies analysis
    if 'investee_company_beid' in tavily_df.columns and 'investee_company_beid' in firecrawl_df.columns:
        tavily_beids = set(tavily_df['investee_company_beid'])
        firecrawl_beids = set(firecrawl_df['investee_company_beid'])
        common_beids = tavily_beids.intersection(firecrawl_beids)
        
        print(f"\nCompany Overlap:")
        print(f"   • Common companies: {len(common_beids)}")
        print(f"   • Tavily only: {len(tavily_beids - firecrawl_beids)}")
        print(f"   • Firecrawl only: {len(firecrawl_beids - tavily_beids)}")
    
    # AI-related insights
    ai_columns = ['AIProduct', 'AITech', 'Genuine']
    print(f"\nAI-Related Metrics:")
    for col in ai_columns:
        if col in tavily_df.columns and col in firecrawl_df.columns:
            tavily_pct = (tavily_df[col] == 1).mean() * 100
            firecrawl_pct = (firecrawl_df[col] == 1).mean() * 100
            diff = firecrawl_pct - tavily_pct
            print(f"   • {col}: Tavily {tavily_pct:.1f}% vs Firecrawl {firecrawl_pct:.1f}% ({diff:+.1f}%)")

def main():
    """Main comparison function"""
    print("TAVILY vs FIRECRAWL DATASET COMPARISON")
    print("=" * 60)
    
    # Target columns to analyze
    target_columns = [
        'ipo', 'ipo_date', 'ma', 'ma_date',
        'CS Degree', 'Top10 University', 'Prior Success IPO', 'Prior Success MA',
        'NumberofExecutives', 'AIProduct', 'AITech', 'Partner', 'Genuine'
    ]
    
    # Load datasets
    tavily_df, firecrawl_df = load_datasets()
    if tavily_df is None or firecrawl_df is None:
        return
    
    # Check column availability
    tavily_available, firecrawl_available = analyze_column_availability(tavily_df, firecrawl_df, target_columns)
    
    # Binary columns comparison
    binary_columns = ['ipo', 'ma', 'Top10 University', 'Prior Success IPO', 'Prior Success MA', 
                     'AIProduct', 'AITech', 'Partner', 'Genuine']
    compare_binary_columns(tavily_df, firecrawl_df, binary_columns)
    
    # Categorical columns comparison
    categorical_columns = ['CS Degree']
    compare_categorical_columns(tavily_df, firecrawl_df, categorical_columns)
    
    # Continuous columns comparison
    continuous_columns = ['NumberofExecutives']
    compare_continuous_columns(tavily_df, firecrawl_df, continuous_columns)
    
    # Generate insights
    generate_summary_insights(tavily_df, firecrawl_df)
    
    print(f"\nComparison complete!")

if __name__ == "__main__":
    main() 