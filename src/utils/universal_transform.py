"""
Universal CSV Transformation Script
Transforms nested objects in CSV files to expanded format.
Supports: founders, executives, products, technologies, partners.

full_founder.csv → full_founder_expanded_clean.csv
full_technology.csv → full_technology_expanded_clean.csv
full_product.csv → full_product_expanded_clean.csv
full_executive.csv → full_executive_expanded_clean.csv
full_partner.csv → full_partner_expanded_clean.csv
"""

import csv
import sys
import os
from src.basedata_control import Founder, Executive, Products, Technology, Partner

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def is_top10_university(university_name):
    """Check if a university belongs to the top 10 universities list."""
    if not university_name or university_name.strip() == "" or university_name.lower() in ["unknown", "others"]:
        return 0
    
    uni_normalized = university_name.lower().strip()
    top10_universities = {
        "mit": ["mit", "massachusetts institute of technology"],
        "cambridge": ["university of cambridge", "cambridge university", "cambridge", "cam"],
        "oxford": ["university of oxford", "oxford university", "oxford"],
        "harvard": ["harvard university", "harvard", "harvard college"],
        "stanford": ["stanford university", "stanford"],
        "imperial": ["imperial college london", "imperial college", "imperial"],
        "eth": ["eth zurich", "eth", "swiss federal institute of technology"],
        "nus": ["national university of singapore", "nus", "national university of singapore (nus)"],
        "ucl": ["university college london", "ucl", "university college london (ucl)"],
        "berkeley": ["university of california, berkeley", "uc berkeley", "berkeley", 
                    "university of california berkeley", "university of california, berkeley (ucb)", "ucb"]
    }
    
    for variations in top10_universities.values():
        for variation in variations:
            if variation in uni_normalized:
                return 1
    return 0

def get_model_config(data_type):
    """Get model class and column name for a data type."""
    configs = {
        'founder': (Founder, 'founders'),
        'executive': (Executive, 'executives'),
        'product': (Products, 'products'),
        'technology': (Technology, 'technologies'),
        'partner': (Partner, 'partners')
    }
    return configs.get(data_type, (Founder, 'founders'))

def parse_objects(object_str, class_name):
    """Parse objects with balanced parentheses handling."""
    if not object_str or object_str.strip() == "":
        return []
    
    try:
        objects_data = []
        class_pattern = f'{class_name.__name__}('
        
        i = 0
        while i < len(object_str):
            class_match = object_str.find(class_pattern, i)
            if class_match == -1:
                break
            
            # Find matching closing parenthesis
            paren_count = 0
            start_pos = class_match + len(class_pattern)
            j = start_pos - 1
            
            while j < len(object_str):
                if object_str[j] == '(':
                    paren_count += 1
                elif object_str[j] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        obj_content = object_str[start_pos:j]
                        obj_dict = parse_object_content(obj_content)
                        if obj_dict:
                            objects_data.append(obj_dict)
                        i = j + 1
                        break
                j += 1
            else:
                break
        
        return objects_data
    except Exception as e:
        print(f"Error parsing {class_name.__name__} objects: {e}")
        return []

def parse_object_content(obj_str):
    """Parse key-value pairs from object content."""
    obj_dict = {}
    pairs = []
    current_pair = ""
    paren_level = 0
    quote_char = None
    
    for char in obj_str:
        if char in ['"', "'"]:
            if quote_char is None:
                quote_char = char
            elif quote_char == char:
                quote_char = None
        elif char == '(' and quote_char is None:
            paren_level += 1
        elif char == ')' and quote_char is None:
            paren_level -= 1
        elif char == ',' and paren_level == 0 and quote_char is None:
            pairs.append(current_pair.strip())
            current_pair = ""
            continue
        current_pair += char
    
    if current_pair.strip():
        pairs.append(current_pair.strip())
    
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').replace('-', '').isdigit():
                try:
                    value = float(value)
                except ValueError:
                    pass
            
            obj_dict[key] = value
    
    return obj_dict

def find_data_column(header, csv_reader, model_class, column_name):
    """Find the column containing the target data."""
    target_index = None
    
    # Special handling for technology and product files with mixed structure
    if model_class.__name__ in ['Technology', 'Products', 'Executive', 'Partner']:
        sample_rows = []
        for _ in range(3):
            try:
                row = next(csv_reader)
                sample_rows.append(row)
            except StopIteration:
                break
        
        # Look for the appropriate data pattern
        if model_class.__name__ == 'Technology':
            search_pattern = 'technologies='
        elif model_class.__name__ == 'Products':
            search_pattern = 'products='
        elif model_class.__name__ == 'Executive':
            search_pattern = 'executives='
        else:  # Partner
            search_pattern = 'partners='
        
        for sample_row in sample_rows:
            for i, cell in enumerate(sample_row):
                if search_pattern in str(cell):
                    target_index = i
                    actual_column = header[i].strip().replace('\ufeff', '').strip()
                    print(f"Found {model_class.__name__.lower()} data in column {i}: '{actual_column}'")
                    break
            if target_index is not None:
                break
        
        return target_index
    
    # Standard column detection
    for i, field in enumerate(header):
        clean_field = field.strip().replace('\ufeff', '').strip()
        if clean_field == column_name:
            return i
    
    return None

def extract_technology_data(objects_str):
    """Extract technologies data from mixed content."""
    if 'technologies=' not in objects_str:
        return objects_str
    
    tech_start = objects_str.find('technologies=')
    if tech_start == -1:
        return ""
    
    bracket_count = 0
    i = tech_start + len('technologies=')
    if i < len(objects_str) and objects_str[i] == '[':
        bracket_count = 1
        i += 1
        while i < len(objects_str) and bracket_count > 0:
            if objects_str[i] == '[':
                bracket_count += 1
            elif objects_str[i] == ']':
                bracket_count -= 1
            i += 1
        return objects_str[tech_start:i]
    
    return ""

def extract_product_data(objects_str):
    """Extract products data from mixed content."""
    if 'products=' not in objects_str:
        return objects_str
    
    prod_start = objects_str.find('products=')
    if prod_start == -1:
        return ""
    
    bracket_count = 0
    i = prod_start + len('products=')
    if i < len(objects_str) and objects_str[i] == '[':
        bracket_count = 1
        i += 1
        while i < len(objects_str) and bracket_count > 0:
            if objects_str[i] == '[':
                bracket_count += 1
            elif objects_str[i] == ']':
                bracket_count -= 1
            i += 1
        return objects_str[prod_start:i]
    
    return ""

def extract_executive_data(objects_str):
    """Extract executives data from mixed content."""
    if 'executives=' not in objects_str:
        return objects_str
    
    exec_start = objects_str.find('executives=')
    if exec_start == -1:
        return ""
    
    bracket_count = 0
    i = exec_start + len('executives=')
    if i < len(objects_str) and objects_str[i] == '[':
        bracket_count = 1
        i += 1
        while i < len(objects_str) and bracket_count > 0:
            if objects_str[i] == '[':
                bracket_count += 1
            elif objects_str[i] == ']':
                bracket_count -= 1
            i += 1
        return objects_str[exec_start:i]
    
    return ""

def extract_partner_data(objects_str):
    """Extract partners data from mixed content."""
    if 'partners=' not in objects_str:
        return objects_str
    
    partner_start = objects_str.find('partners=')
    if partner_start == -1:
        return ""
    
    bracket_count = 0
    i = partner_start + len('partners=')
    if i < len(objects_str) and objects_str[i] == '[':
        bracket_count = 1
        i += 1
        while i < len(objects_str) and bracket_count > 0:
            if objects_str[i] == '[':
                bracket_count += 1
            elif objects_str[i] == ']':
                bracket_count -= 1
            i += 1
        return objects_str[partner_start:i]
    
    return ""

def extract_embedded_data(objects_str, model_class):
    """Extract specific data from mixed content."""
    if model_class.__name__ == 'Technology':
        return extract_technology_data(objects_str)
    elif model_class.__name__ == 'Products':
        return extract_product_data(objects_str)
    elif model_class.__name__ == 'Executive':
        return extract_executive_data(objects_str)
    elif model_class.__name__ == 'Partner':
        return extract_partner_data(objects_str)
    return objects_str

def transform_csv(data_type):
    """Transform CSV file for the specified data type."""
    input_file = f"full_{data_type}.csv"
    output_file = f"full_{data_type}_expanded_clean.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    model_class, column_name = get_model_config(data_type)
    print(f"Processing {input_file} using {model_class.__name__} model")
    
    with open(input_file, 'r', encoding='utf-8', newline='') as infile:
        csv_reader = csv.reader(infile)
        
        try:
            header = next(csv_reader)
        except StopIteration:
            print("Error: Input file is empty.")
            return
        
        target_index = find_data_column(header, csv_reader, model_class, column_name)
        
        if target_index is None:
            # Reset and try standard detection
            infile.seek(0)
            csv_reader = csv.reader(infile)
            header = next(csv_reader)
            
            for i, field in enumerate(header):
                clean_field = field.strip().replace('\ufeff', '').strip()
                if clean_field == column_name:
                    target_index = i
                    break
        
        if target_index is None:
            cleaned_columns = [field.strip().replace('\ufeff', '').strip() for field in header]
            print(f"Error: '{column_name}' column not found. Available: {cleaned_columns}")
            return
        
        # Reset reader if needed
        if model_class.__name__ in ['Technology', 'Products', 'Executive', 'Partner']:
            infile.seek(0)
            csv_reader = csv.reader(infile)
            next(csv_reader)  # Skip header
        
        # Prepare output
        model_fields = list(model_class.__annotations__.keys())
        if model_class.__name__ == 'Founder':
            model_fields = model_fields + ['is_top10_university']
        
        output_headers = ['investee_company_beid', 'investee_company_name'] + model_fields
        output_rows = []
        
        # Process rows
        for line_num, row_data in enumerate(csv_reader, start=2):
            try:
                if len(row_data) <= target_index:
                    continue
                
                company_beid = row_data[0].strip() if row_data else ""
                company_name = row_data[1].strip() if len(row_data) > 1 else ""
                objects_str = row_data[target_index].strip() if target_index < len(row_data) else ""
                
                # Special handling for technology and product data embedded in mixed columns
                if model_class.__name__ in ['Technology', 'Products', 'Executive', 'Partner']:
                    if model_class.__name__ == 'Technology':
                        search_pattern = 'technologies='
                    elif model_class.__name__ == 'Products':
                        search_pattern = 'products='
                    elif model_class.__name__ == 'Executive':
                        search_pattern = 'executives='
                    else:  # Partner
                        search_pattern = 'partners='
                    
                    if search_pattern in objects_str:
                        objects_str = extract_embedded_data(objects_str, model_class)
                
                objects = parse_objects(objects_str, model_class)
                
                if not objects:
                    empty_row = [company_beid, company_name] + [""] * len(model_fields)
                    if model_class.__name__ == 'Founder':
                        empty_row[-1] = 0
                    output_rows.append(empty_row)
                else:
                    for obj in objects:
                        output_row = [company_beid, company_name]
                        
                        for field in model_fields[:-1] if model_class.__name__ == 'Founder' else model_fields:
                            value = obj.get(field, "")
                            if isinstance(value, str):
                                value = value.strip().strip('"').strip("'")
                            output_row.append(value)
                        
                        if model_class.__name__ == 'Founder':
                            graduated_university = obj.get('graduated_university', "")
                            if isinstance(graduated_university, str):
                                graduated_university = graduated_university.strip().strip('"').strip("'")
                            top10_flag = is_top10_university(graduated_university)
                            output_row.append(top10_flag)
                        
                        output_rows.append(output_row)
            
            except Exception as e:
                print(f"Error processing row {line_num}: {e}")
                continue
    
    # Write output
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(output_headers)
        writer.writerows(output_rows)
    
    print(f"✅ Transformation complete: {output_file}")
    print(f"📊 Total {model_class.__name__} objects extracted: {len(output_rows)}")
    
    if model_class.__name__ == 'Founder':
        top10_count = sum(1 for row in output_rows if len(row) > len(model_fields) and row[-1] == 1)
        print(f"🎓 Founders from top 10 universities: {top10_count} out of {len(output_rows)} ({top10_count/len(output_rows)*100:.1f}%)")

if __name__ == "__main__":
    # Change this to process different data types
    data_type = sys.argv[1] if len(sys.argv) > 1 else 'product'
    transform_csv(data_type) 