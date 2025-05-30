"""
Compare Tavily and Firecrawl datasets against S&P data as ground truth
Focus on IPO, IPO Date, M&A, and M&A Date accuracy
"""
import pandas as pd
import numpy as np
from datetime import datetime

def load_all_datasets():
    """Load all three datasets"""
    datasets = {}
    
    try:
        datasets['sp'] = pd.read_csv('merged_output/s&p_final_combined_data.csv')
    except FileNotFoundError:
        print("ERROR: S&P dataset not found at merged_output/s&p_final_combined_data.csv")
        return None
    
    try:
        datasets['tavily'] = pd.read_csv('outputs/merged_data.csv')
    except FileNotFoundError:
        print("ERROR: Tavily dataset not found")
        return None
    
    try:
        datasets['firecrawl'] = pd.read_csv('outputs/merged_firecrawl_data.csv')
    except FileNotFoundError:
        print("ERROR: Firecrawl dataset not found")
        return None
    
    return datasets

def find_common_companies(datasets):
    """Find companies that exist in all three datasets"""
    sp_beids = set(datasets['sp']['investee_company_beid'])
    tavily_beids = set(datasets['tavily']['investee_company_beid'])
    firecrawl_beids = set(datasets['firecrawl']['investee_company_beid'])
    
    # Find intersection of all three
    common_beids = sp_beids.intersection(tavily_beids).intersection(firecrawl_beids)
    
    return sorted(list(common_beids))

def standardize_transaction_data(datasets, common_beids):
    """Standardize transaction data across all datasets"""
    
    # Filter to common companies and sort by BEID
    for name, df in datasets.items():
        datasets[name] = df[df['investee_company_beid'].isin(common_beids)].sort_values('investee_company_beid').reset_index(drop=True)
    
    # Standardize column names and create comparison DataFrame
    comparison_data = []
    
    for beid in common_beids:
        # Get data for this company from each dataset
        sp_row = datasets['sp'][datasets['sp']['investee_company_beid'] == beid].iloc[0]
        tavily_row = datasets['tavily'][datasets['tavily']['investee_company_beid'] == beid].iloc[0]
        firecrawl_row = datasets['firecrawl'][datasets['firecrawl']['investee_company_beid'] == beid].iloc[0]
        
        # Extract transaction data
        row_data = {
            'investee_company_beid': beid,
            'company_name': sp_row.get('investee_company_name', ''),
            
            # IPO data
            'sp_ipo': 1 if sp_row.get('Is_IPO', False) else 0,
            'sp_ipo_date': sp_row.get('IPO_Date', ''),
            'tavily_ipo': tavily_row.get('ipo', 0),
            'tavily_ipo_date': tavily_row.get('ipo_date', ''),
            'firecrawl_ipo': firecrawl_row.get('ipo', 0),
            'firecrawl_ipo_date': firecrawl_row.get('ipo_date', ''),
            
            # M&A data
            'sp_ma': 1 if sp_row.get('Is_MA', False) else 0,
            'sp_ma_date': sp_row.get('Deal_Completion_Date', ''),
            'tavily_ma': tavily_row.get('ma', 0),
            'tavily_ma_date': tavily_row.get('ma_date', ''),
            'firecrawl_ma': firecrawl_row.get('ma', 0),
            'firecrawl_ma_date': firecrawl_row.get('ma_date', ''),
        }
        
        comparison_data.append(row_data)
    
    return pd.DataFrame(comparison_data)

def calculate_accuracy_metrics(comparison_df):
    """Calculate accuracy metrics using S&P as ground truth"""
    
    print("ACCURACY ANALYSIS (S&P as Ground Truth)")
    print("=" * 60)
    
    total_companies = len(comparison_df)
    
    # IPO Accuracy
    tavily_ipo_matches = (comparison_df['sp_ipo'] == comparison_df['tavily_ipo']).sum()
    firecrawl_ipo_matches = (comparison_df['sp_ipo'] == comparison_df['firecrawl_ipo']).sum()
    
    print(f"\nIPO Detection Accuracy:")
    print(f"- Tavily: {tavily_ipo_matches}/{total_companies} ({tavily_ipo_matches/total_companies*100:.1f}%)")
    print(f"- Firecrawl: {firecrawl_ipo_matches}/{total_companies} ({firecrawl_ipo_matches/total_companies*100:.1f}%)")
    
    # M&A Accuracy
    tavily_ma_matches = (comparison_df['sp_ma'] == comparison_df['tavily_ma']).sum()
    firecrawl_ma_matches = (comparison_df['sp_ma'] == comparison_df['firecrawl_ma']).sum()
    
    print(f"\nM&A Detection Accuracy:")
    print(f"- Tavily: {tavily_ma_matches}/{total_companies} ({tavily_ma_matches/total_companies*100:.1f}%)")
    print(f"- Firecrawl: {firecrawl_ma_matches}/{total_companies} ({firecrawl_ma_matches/total_companies*100:.1f}%)")
    
    # Overall Transaction Accuracy (both IPO and M&A correct)
    tavily_both_correct = ((comparison_df['sp_ipo'] == comparison_df['tavily_ipo']) & 
                          (comparison_df['sp_ma'] == comparison_df['tavily_ma'])).sum()
    firecrawl_both_correct = ((comparison_df['sp_ipo'] == comparison_df['firecrawl_ipo']) & 
                             (comparison_df['sp_ma'] == comparison_df['firecrawl_ma'])).sum()
    
    print(f"\nOverall Transaction Accuracy (Both IPO & M&A Correct):")
    print(f"- Tavily: {tavily_both_correct}/{total_companies} ({tavily_both_correct/total_companies*100:.1f}%)")
    print(f"- Firecrawl: {firecrawl_both_correct}/{total_companies} ({firecrawl_both_correct/total_companies*100:.1f}%)")
    
    return {
        'tavily_ipo_accuracy': tavily_ipo_matches/total_companies*100,
        'firecrawl_ipo_accuracy': firecrawl_ipo_matches/total_companies*100,
        'tavily_ma_accuracy': tavily_ma_matches/total_companies*100,
        'firecrawl_ma_accuracy': firecrawl_ma_matches/total_companies*100,
        'tavily_overall_accuracy': tavily_both_correct/total_companies*100,
        'firecrawl_overall_accuracy': firecrawl_both_correct/total_companies*100
    }

def create_detailed_comparison_table(comparison_df):
    """Create detailed comparison table and save to CSV"""
    
    # Select key columns for the output
    output_df = comparison_df[[
        'investee_company_beid', 'company_name',
        'sp_ipo', 'tavily_ipo', 'firecrawl_ipo',
        'sp_ipo_date', 'tavily_ipo_date', 'firecrawl_ipo_date',
        'sp_ma', 'tavily_ma', 'firecrawl_ma',
        'sp_ma_date', 'tavily_ma_date', 'firecrawl_ma_date'
    ]].copy()
    
    # Add accuracy flags
    output_df['tavily_ipo_correct'] = (output_df['sp_ipo'] == output_df['tavily_ipo']).astype(int)
    output_df['firecrawl_ipo_correct'] = (output_df['sp_ipo'] == output_df['firecrawl_ipo']).astype(int)
    output_df['tavily_ma_correct'] = (output_df['sp_ma'] == output_df['tavily_ma']).astype(int)
    output_df['firecrawl_ma_correct'] = (output_df['sp_ma'] == output_df['firecrawl_ma']).astype(int)
    
    # Save to CSV
    output_file = 'outputs/sp_validation_comparison.csv'
    output_df.to_csv(output_file, index=False)
    
    print(f"\nDetailed comparison saved to: {output_file}")
    
    return output_df

def generate_summary_insights(accuracy_metrics, comparison_df):
    """Generate key insights from the S&P validation"""
    
    print("\nKEY VALIDATION INSIGHTS")
    print("=" * 30)
    
    insights = []
    
    # Overall accuracy comparison
    if accuracy_metrics['tavily_overall_accuracy'] > accuracy_metrics['firecrawl_overall_accuracy']:
        diff = accuracy_metrics['tavily_overall_accuracy'] - accuracy_metrics['firecrawl_overall_accuracy']
        insights.append(f"Tavily shows higher overall transaction accuracy (+{diff:.1f}%)")
    elif accuracy_metrics['firecrawl_overall_accuracy'] > accuracy_metrics['tavily_overall_accuracy']:
        diff = accuracy_metrics['firecrawl_overall_accuracy'] - accuracy_metrics['tavily_overall_accuracy']
        insights.append(f"Firecrawl shows higher overall transaction accuracy (+{diff:.1f}%)")
    
    # IPO-specific insights
    if accuracy_metrics['tavily_ipo_accuracy'] > accuracy_metrics['firecrawl_ipo_accuracy']:
        diff = accuracy_metrics['tavily_ipo_accuracy'] - accuracy_metrics['firecrawl_ipo_accuracy']
        insights.append(f"Tavily more accurate for IPO detection (+{diff:.1f}%)")
    elif accuracy_metrics['firecrawl_ipo_accuracy'] > accuracy_metrics['tavily_ipo_accuracy']:
        diff = accuracy_metrics['firecrawl_ipo_accuracy'] - accuracy_metrics['tavily_ipo_accuracy']
        insights.append(f"Firecrawl more accurate for IPO detection (+{diff:.1f}%)")
    
    # M&A-specific insights
    if accuracy_metrics['tavily_ma_accuracy'] > accuracy_metrics['firecrawl_ma_accuracy']:
        diff = accuracy_metrics['tavily_ma_accuracy'] - accuracy_metrics['firecrawl_ma_accuracy']
        insights.append(f"Tavily more accurate for M&A detection (+{diff:.1f}%)")
    elif accuracy_metrics['firecrawl_ma_accuracy'] > accuracy_metrics['tavily_ma_accuracy']:
        diff = accuracy_metrics['firecrawl_ma_accuracy'] - accuracy_metrics['tavily_ma_accuracy']
        insights.append(f"Firecrawl more accurate for M&A detection (+{diff:.1f}%)")
    
    # Data quality assessment
    total_companies = len(comparison_df)
    sp_total_transactions = (comparison_df['sp_ipo'] == 1).sum() + (comparison_df['sp_ma'] == 1).sum()
    
    insights.append(f"S&P ground truth shows {sp_total_transactions} total transactions across {total_companies} companies")
    
    print("Key Findings:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print(f"\nValidation Summary:")
    print(f"- This analysis validates web scraping accuracy against authoritative S&P data")
    print(f"- Results show which methodology (Tavily vs Firecrawl) is more reliable")
    print(f"- Accuracy differences highlight the importance of data source validation")

def main():
    """Main validation function"""
    print("S&P GROUND TRUTH VALIDATION ANALYSIS")
    print("=" * 50)
    
    # Load all datasets
    datasets = load_all_datasets()
    if datasets is None:
        return
    
    # Find common companies
    common_beids = find_common_companies(datasets)
    if len(common_beids) == 0:
        print("ERROR: No common companies found across all three datasets")
        return
    
    # Create standardized comparison
    comparison_df = standardize_transaction_data(datasets, common_beids)
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(comparison_df)
    
    # Create detailed output table
    create_detailed_comparison_table(comparison_df)
    
    # Generate insights
    generate_summary_insights(accuracy_metrics, comparison_df)
    
    print(f"\nValidation analysis complete!")

if __name__ == "__main__":
    main() 