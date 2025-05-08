import pandas as pd
import asyncio
import os
from dotenv import load_dotenv
from llm_analyst import llm_articles
from langchain_openai import ChatOpenAI
from prompts import *
from basedata import *

async def process_single_batch():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    xls_path = os.getenv("XLS_PATH")
    
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    if not xls_path:
        raise ValueError("Please set XLS_PATH environment variable")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=api_key,
        max_tokens=200
    )
    
    # Select prompt from the additional_prompts
    prompt = AUTOMATION_DEPTH  # Change to whichever prompt you need
    
    # Load data
    df = pd.read_excel(xls_path)
    articles = df.to_dict('records')
    
    # Define parameters
    start_index = 4300
    batch_size = 50
    output_path = f"batch_{start_index}_{batch_size}_results.json"
    
    # Extract the batch
    batch = articles[start_index:start_index + batch_size]
    
    # Process the batch
    # Set batch_size equal to the length of our batch to ensure it's processed as one unit
    results = await llm_articles(batch, llm, prompt, batch_size=len(batch), output_path=output_path)
    
    print(f"Processed batch starting at index {start_index}. Results saved to {output_path}")
    return results

if __name__ == "__main__":
    asyncio.run(process_single_batch()) 
    print("done")