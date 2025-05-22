import pandas as pd
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llm_analyst import llm_articles, map_response
from prompt.prompts import *
from src.basedata import *

# Load environment variables
load_dotenv()

def check_empty_fields(row, prompt):
    """
    Check if any fields in the response class for the given prompt are empty.
    
    Args:
        row: DataFrame row
        prompt: Prompt name string
    
    Returns:
        bool: True if any field is empty, False otherwise
    """
    # Get the expected fields from the response class
    response_class = map_response[prompt]
    expected_fields = response_class.schema()['properties'].keys()
    
    # Check if any of the expected fields are empty in the row
    for field in expected_fields:
        if field in row and (pd.isna(row[field]) or row[field] == ""):
            return True
    
    return False

async def fill_empty_fields():
    """Main function to fill empty fields in abnormal.csv"""
    
    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=api_key,
        max_tokens=300
    )
    
    # Read the abnormal.csv file
    print("Reading abnormal.csv...")
    df = pd.read_csv("abnormal_filled.csv")
    
    # Define the output file
    output_path = "abnormal_filled.csv"
    
    # List of prompts to process
    prompt_list = base_prompt + additional_prompt
    
    # Process each prompt
    for prompt in prompt_list:
        print(f"Processing prompt: {prompt[:50]}...")
        
        # Identify rows that need fixing for this prompt
        rows_to_fix = []
        indices_to_fix = []
        
        # Iterate through rows and check if any field is empty for the current prompt
        for i, row in df.iterrows():
            if check_empty_fields(row, prompt):
                # Add the row only once regardless of how many empty fields it has
                rows_to_fix.append(row.to_dict())
                indices_to_fix.append(i)
        
        print(f"Found {len(rows_to_fix)} rows with empty fields for this prompt")
        
        if not rows_to_fix:
            continue
        
        # Process in batches of 50
        batch_size = 50
        for i in range(0, len(rows_to_fix), batch_size):
            batch = rows_to_fix[i:i + batch_size]
            batch_indices = indices_to_fix[i:i + batch_size]
            
            print(f"Processing batch {i//batch_size + 1} of {(len(rows_to_fix) + batch_size - 1)//batch_size}")
            
            # Process the batch
            batch_output_path = f"temp_batch.csv"
            updated_batch = await llm_articles(batch, llm, prompt, batch_size=len(batch), output_path=batch_output_path)
            
            # Update the DataFrame with the new results
            for j, updated_row in enumerate(updated_batch):
                original_index = batch_indices[j]
                
                # Only update the fields from the response class
                response_class = map_response[prompt]
                expected_fields = response_class.schema()['properties'].keys()
                
                for field in expected_fields:
                    if field in updated_row and not pd.isna(updated_row[field]) and updated_row[field] != "":
                        df.at[original_index, field] = updated_row[field]
            
            # Clean up temporary file
            if os.path.exists(batch_output_path):
                os.remove(batch_output_path)
        
        # Save the updated DataFrame after processing each prompt
        df.to_csv(output_path, index=False)
        print(f"Saved updated data to {output_path} after processing prompt")
    
    print(f"All prompts processed. Final results saved to {output_path}")
    return df

if __name__ == "__main__":
    asyncio.run(fill_empty_fields()) 