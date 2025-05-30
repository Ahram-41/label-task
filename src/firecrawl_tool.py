from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser

from src.prompt import *
from src.basedata_control import *
import pandas as pd
from typing import List, Dict
import json
import time
import asyncio
from tqdm import tqdm
from langchain_openai import ChatOpenAI

load_dotenv()



firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

kwargs = {
    "max_depth": 10,  # Number of research iterations
    "time_limit": 300,  # Time limit in secondss
    "max_urls": 8  # Maximum URLs to analyze
}
task_map={"f_executive": FIRE_EXECUTIVE_PROMPT, "f_ipo_mna": FIRE_IPO_MNA_PROMPT, "f_founder": FIRE_FOUNDER_PROMPT, "f_product": FIRE_PRODUCT_PROMPT, "f_technology": FIRE_TECHNOLOGY_PROMPT, "f_partner": FIRE_PARTNER_PROMPT}

task_2_llm = {
    "f_executive": COMPANY_EXECUTIVES,
    "f_ipo_mna": IPO_MNA,
    "f_founder": FOUNDERS_BACKGROUND,
    "f_product": AI_PRODUCTS,
    "f_technology": AI_TECHNOLOGY,
    "f_partner": AI_PARTNER
}
def on_activity(activity):
    print(f"[{activity['type']}] {activity['message']}")

def process_csv(task):
    ans=[]
    df = pd.read_csv("others/distinct_selected_companies.csv")
    df = df.head(20)
    for index, row in df.iterrows():
        start_time = time.time()
        company_info = row.to_dict()
        prompt = PromptTemplate(
            template=task_map[task],
            input_variables=["company_name", "description"],
        ).format(
            company_name=company_info["investee_company_name"],
            description=company_info["investee_company_long_business_d"]
        )
        # Run deep research
        try:
            results = firecrawl.deep_research(query=prompt, on_activity=on_activity, **kwargs)
            # Access research findings
            print(f"Final Analysis: {results['data']['finalAnalysis']}")
            print(f"Sources: {len(results['data']['sources'])} references")
            company_info = {**company_info,
                            "success_search":results['success'],
                            "finalAnalysis":results['data']['finalAnalysis'],
                            "source":results['data']['sources']}

            print(company_info)
        except Exception as e:
            print(f"Error processing {company_info['investee_company_name']}: {str(e)}")
            company_info = {**company_info,
                            "success_search":False,
                            "finalAnalysis":None,
                            "source":None}
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time for {company_info['investee_company_name']}: {processing_time:.2f} seconds")
        company_info['processing_time'] = processing_time
        ans.append(company_info)
    df_processed = pd.DataFrame(ans)
    df_processed.to_csv(f"others/distinct_selected_companies_firecrawl_{task}.csv", index=False, encoding='utf-8-sig')
    return df_processed

map_response={
    IPO_MNA: IPO_MnA_Response,
    FOUNDERS_BACKGROUND: Founders_Background_Response,
    AI_PRODUCTS: Products,
    AI_TECHNOLOGY: Technology,
    AI_PARTNER: Partner
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
        input_variables=["company_name", "description"],
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
        
        batch_inputs = [{"company_name": article['investee_company_name'], "description": article['finalAnalysis']} for article in batch]
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
    llm = ChatOpenAI(
                model_name="gpt-4o",
                openai_api_key=api_key,
                max_tokens=300 # limit the output length to 200 tokens
            )
    def run_analysis(articles: List[Dict], prompt: str, output_path: str = None):
        ans = asyncio.run(llm_articles(articles, llm, prompt, output_path=output_path))
        return ans
    start_index=0
    TASK_NAME = ["f_executive", "f_ipo_mna", "f_founder", "f_product", "f_technology", "f_partner"]
    for task in TASK_NAME:
        df = process_csv(task)

    for task in TASK_NAME:
        df = pd.read_csv(f"others/distinct_selected_companies_firecrawl_{task}.csv")
        articles = df.to_dict('records')
        output_path = f"outputs/llm_firecrawl_output_{task}.csv"
        # If starting from a specific index, slice the articles list
        append_mode = start_index > 0
        if append_mode:
            articles = articles[start_index:]
            print(f"Starting processing from index {start_index} ({len(articles)} articles remaining)")
            
        prompt = task_2_llm[task]
        # for i, prompt in enumerate(tqdm(prompts)):
        articles = run_analysis(articles, prompt, output_path)
        # save the articles to a csv file
        df_articles = pd.DataFrame(articles)
        df_articles.to_csv(output_path, index=False, encoding='utf-8-sig')

