import csv
import os
import pandas as pd
from basedata import *

# Get all field names from basedata classes
def get_all_basedata_fields():
    fields = []
    # Get all classes defined in basedata.py
    basedata_classes = [
        Core_or_General_AI_Application_Response,
        Is_Data_Centric_Response,
        Is_Niche_or_Broad_Market_A_Response,
        Is_Niche_or_Broad_Market_B_Response,
        Is_Product_or_Platform_Response,
        AI_startup_type_Response,
        Is_Data_Centric_Response2,
        Is_Niche_or_Broad_Market_Response2,
        Is_Product_or_Platform_Response2,
        AI_startup_type_Response3,
        Developer_or_Integrator_Response,
        AI_Native_or_Augmented_Response,
        Automation_Depth_Response
    ]
    
    # Extract field names from each class
    for cls in basedata_classes:
        for field_name in cls.__fields__:
            fields.append(field_name)
    
    return fields

def main():
    input_file = "./full_sample_label_all.csv"
    normal_file = "normal.csv"
    abnormal_file = "abnormal.csv"
    
    # Get all fields from basedata
    basedata_fields = get_all_basedata_fields()
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return
    
    try:
        df = pd.read_csv(input_file)
        
        # Filter basedata fields that exist in the CSV
        existing_fields = [field for field in basedata_fields if field in df.columns]
        
        if not existing_fields:
            print("No basedata fields found in the CSV file")
            return
        
        # Check which rows have all basedata fields filled
        normal_rows = df.copy()
        abnormal_rows = df.copy()
        
        # A row is normal if none of the basedata fields are empty
        mask = df[existing_fields].notna().all(axis=1)
        
        normal_rows = df[mask]
        abnormal_rows = df[~mask]
        
        # Save to separate files
        normal_rows.to_csv(normal_file, index=False)
        abnormal_rows.to_csv(abnormal_file, index=False)
        
        print(f"Split complete:")
        print(f"  - Normal rows: {len(normal_rows)} (saved to {normal_file})")
        print(f"  - Abnormal rows: {len(abnormal_rows)} (saved to {abnormal_file})")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main() 