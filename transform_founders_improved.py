import csv
import re
import sys
import os
import pathlib

# Add the src directory to the path to import the basedata_control module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.basedata_control import Founder, Executive, Products, Technology

def get_model_class_from_filename(filename):
    """Determine the appropriate model class based on the filename"""
    filename_lower = filename.lower()
    
    if 'founder' in filename_lower:
        return Founder, 'founders'
    elif 'executive' in filename_lower:
        return Executive, 'executives'  
    elif 'product' in filename_lower:
        return Products, 'products'
    elif 'tech' in filename_lower or 'technology' in filename_lower:
        return Technology, 'technologies'
    else:
        # Default to Founder if we can't determine
        print(f"Warning: Could not determine object type from filename '{filename}'. Defaulting to Founder.")
        return Founder, 'founders'

# Function to safely parse objects from string
def parse_objects(object_str, class_name):
    """Parse objects of the given class from a string"""
    if not object_str or object_str == "":
        return []
    
    try:
        # Use regex to extract objects
        pattern = fr'{class_name.__name__}\(([^)]+)\)'
        objects_data = []
        
        for match in re.finditer(pattern, object_str):
            obj_str = match.group(1)
            # Parse each key-value pair
            obj_dict = {}
            for pair in obj_str.split(', '):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    # Remove quotes for string values
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    # Convert numeric values
                    elif value.isdigit():
                        value = int(value)
                    obj_dict[key] = value
            objects_data.append(obj_dict)
        
        return objects_data
    except Exception as e:
        print(f"Error parsing {class_name.__name__} objects: {e}")
        return []

# Read the original CSV
name = 'technology'
input_file = f"{name}.csv"  # This can be changed based on what you're processing
output_file = f'{name}_expanded_clean.csv'

# Determine the model class based on the input filename
model_class, column_name = get_model_class_from_filename(input_file)
print(f"Processing {input_file} using {model_class.__name__} model, looking for '{column_name}' column")

# Open the input file and read all rows
with open(input_file, 'r', encoding='utf-8') as infile:
    # Read as list of lines to handle potential CSV parsing issues
    lines = infile.readlines()
    
    # Skip empty lines
    lines = [line for line in lines if line.strip()]
    
    # Parse header
    header = lines[0].strip().split(',')
    
    # Find the index of the 'founders' column
    founders_index = None
    for i, field in enumerate(header):
        if field == 'founders':
            founders_index = i
            break
    
    if founders_index is None:
        print("Error: 'founders' column not found in the header.")
        exit(1)
    
    # Prepare headers for the output CSV
    # Get fields dynamically from the model class
    model_fields = list(model_class.__annotations__.keys())
    
    # We want to keep investee_company_beid and investee_company_name
    output_headers = ['investee_company_beid', 'investee_company_name'] + model_fields
    
    # Prepare rows for output
    output_rows = []
    
    # Process each row (skipping header)
    for line_num, line in enumerate(lines[1:], start=2):
        # Split the row, being careful with quoted fields
        row_data = []
        in_quotes = False
        current_field = ""
        
        for char in line:
            if char == '"' and (len(current_field) == 0 or current_field[-1] != '\\'):
                in_quotes = not in_quotes
                current_field += char
            elif char == ',' and not in_quotes:
                row_data.append(current_field)
                current_field = ""
            else:
                current_field += char
                
        # Add the last field
        if current_field:
            row_data.append(current_field)
        
        # Ensure we have enough columns
        if len(row_data) <= founders_index:
            print(f"Warning: Row {line_num} doesn't have enough columns. Skipping.")
            continue
        
        # Get company info
        company_beid = row_data[0].strip() if len(row_data) > 0 else ""
        company_name = row_data[1].strip().strip('"') if len(row_data) > 1 else ""
        
        # Get founders data
        founders_str = row_data[founders_index].strip() if founders_index < len(row_data) else ""
        objects = parse_objects(founders_str, model_class)
        
        # Create a row for each object
        if not objects:
            # If no objects found, create a row with empty fields
            output_row = [company_beid, company_name] + [""] * len(model_fields)
            output_rows.append(output_row)
        else:
            for obj in objects:
                # Create a row with all possible fields from the model
                output_row = [company_beid, company_name]
                for field in model_fields:
                    output_row.append(obj.get(field, ""))
                output_rows.append(output_row)

    # Write the output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(output_headers)
        writer.writerows(output_rows)

    print(f"Transformation complete. Output saved to {output_file}")
    print(f"Total companies processed: {len(lines) - 1}")
    print(f"Total {model_class.__name__} objects extracted: {len(output_rows)}") 