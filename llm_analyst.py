import pandas as pd
from langchain_openai import ChatOpenAI
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from prompts import *
from basedata import *
from typing import Dict, List
from dotenv import load_dotenv
import os
import asyncio
import time
from tqdm import tqdm

load_dotenv()
map_response = {
    CORE_OR_GENERAL_AI_APPLICATION: Core_or_General_AI_Application_Response,
    IS_DATA_CENTRIC: Is_Data_Centric_Response,
    IS_NICHE_OR_BROAD_MARKET_A: Is_Niche_or_Broad_Market_A_Response,
    IS_NICHE_OR_BROAD_MARKET_B: Is_Niche_or_Broad_Market_B_Response,
    IS_PRODUCT_OR_PLATFORM: Is_Product_or_Platform_Response,
    HARDWARE_OR_FOUNDATION_MODEL: AI_startup_type_Response,
    VERTICAL_OR_HORIZONTAL: Vertical_or_Horizontal_Response,
    DEVELOPER_OR_INTEGRATOR: Developer_or_Integrator_Response,
    AI_NATIVE_OR_AUGMENTED: AI_Native_or_Augmented_Response,
    AUTOMATION_DEPTH: Automation_Depth_Response
}


class LabelTask:
    def __init__(self, prompts: list[str]):
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=api_key,
            max_tokens=300 # limit the output length to 200 tokens
        )
        self.prompts = prompts
        

    def process_xls(self, xls_path: str, output_path: str, start_index: int = 0)->List[Dict]:
        """
        Process CSV file and analyze each article's content with flexible field mapping.
        
        Args:
            xls_path: Path to the xls file
            output_path: Path to save the csv file
            start_index: Index to start processing from (default: 0)
        """
        # Read xls
        df = pd.read_excel(xls_path)
        
        # Pre-process articles to include formatted content
        articles = df.to_dict('records')
        
        # If starting from a specific index, slice the articles list
        append_mode = start_index > 0
        if append_mode:
            articles = articles[start_index:]
            print(f"Starting processing from index {start_index} ({len(articles)} articles remaining)")
            
        for i, prompt in enumerate(tqdm(self.prompts)):
            articles = self.run_analysis(articles, prompt, output_path)
            output_path = output_path.replace('.csv', f'_{i}.csv') # create several check points
        return articles
    
    def run_analysis(self, articles: List[Dict], prompt: str, output_path: str = None):
        ans = asyncio.run(llm_articles(articles, self.llm, prompt, output_path=output_path))
        return ans


def analyze_content(llm, content: str, prompt: str) -> Dict:
    try:
        parser = JsonOutputParser(pydantic_object=map_response[prompt])
        prompt = PromptTemplate(
            template=prompt,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = (prompt | llm | parser)
        result = chain.invoke({"content": content})
        return result
        
    except Exception as e:
        print(f"Error analyzing content: {str(e)}")
        # Create a dictionary directly for error case
        return {
            "reasoning": "Error in analysis"
        }

async def llm_articles(articles: List[Dict], model, prompt_name, batch_size: int = 50, output_path: str = None) -> List[Dict]:
    """
    Process a list of articles using an LLM model with flexible schema handling.
    
    Args:
        articles: List of article dictionaries with 'formatted_content' field
        model: LLM model to use for processing
        batch_size: Number of articles to process in each batch
        output_path: Path to save incremental results
        
    Returns:
        List of processed articles with analysis
    """
    # Configure the output parser
    parser = JsonOutputParser(pydantic_object=map_response[prompt_name])
    
    prompt = PromptTemplate(
        template=prompt_name,
        input_variables=["content"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | model | parser
    expected_keys = map_response[prompt_name].schema()['properties'].keys()
    
    async def batch_process_text(texts: List):
        return await chain.abatch(texts, return_exceptions=True) # return exceptions to handle errors
        
    ans = []
    
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        batch_results = []
        
        batch_inputs = [{"content": article['investee_company_long_business_d']} for article in batch]
        try:
            # Process the batch
            batch_responses = await batch_process_text(batch_inputs)
            # Process responses and add metadata
            for j, response in enumerate(batch_responses):
                article = batch[j]
                print(f"\n=== Article {i + j + 1}/{len(articles)} ===")
                
                try:
                    # If response is an exception, create an error entry
                    if isinstance(response, Exception):
                        # Create a failed entry with all expected fields as None
                        analysis_data = {key: str(response) for key in expected_keys}
                        result_entry = {
                            **article,
                            **analysis_data,
                            'error': str(response),
                            'status': 'failed'
                        }
                    else:
                        # For successful cases, ensure error field exists but is None
                        result_entry = {
                            **article,
                            **response,
                            'error': None,
                            'status': 'success'

                        }
                except Exception as e:
                    # Create a failed entry with all expected fields as None
                    analysis_data = {key: str(e) for key in expected_keys}
                    result_entry = {
                        **article,
                        **analysis_data,
                        'error': f"Error processing response: {str(e)}",
                        'status': 'failed'
                    }
                
                ans.append(result_entry)
                batch_results.append(result_entry)
            
            # Save incremental results after each batch if output_path is provided
            if output_path and batch_results:
                # Append batch results to CSV
                batch_df = pd.DataFrame(batch_results)
                
                # Write batch results to CSV
                file_exists = os.path.exists(output_path)
                batch_df.to_csv(output_path, mode='a', index=False, header= not file_exists)
                
                print(f"Batch {i//batch_size + 1} complete. Results appended to {output_path}")
                
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            with open('error.json', 'a') as f:
                json.dump({'error': str(e), 'prompt': prompt_name, 'batch_size': batch_size, 'index': i}, f, ensure_ascii=False, indent=4)
            time.sleep(5)
            continue

    return ans

if __name__ == "__main__":

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    xls_path = os.getenv("XLS_PATH")
    if not xls_path:
        raise ValueError("Please set XLS_PATH environment variable")
    
    prompt_name = "base_prompt"
    iea = LabelTask(base_prompt)
    # prompt_name = "additional_prompt"
    # iea = LabelTask(additional_prompt)
    output_path = f"full_sample_label_{prompt_name}.csv"
    
    # Start processing from index
    # start_index = 3300
    ans = iea.process_xls(xls_path, output_path)