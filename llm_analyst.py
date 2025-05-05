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
from tqdm import tqdm

load_dotenv()
map_response = {
    CORE_OR_GENERAL_AI_APPLICATION: Core_or_General_AI_Application_Response,
    IS_DATA_CENTRIC: Is_Data_Centric_Response,
    IS_NICHE_OR_BROAD_MARKET: Is_Niche_or_Broad_Market_Response,
    IS_PRODUCT_OR_PLATFORM: Is_Product_or_Platform_Response,
    AI_STARTUP_TYPE2: AI_startup_type_Response,
    IS_DATA_CENTRIC2: Is_Data_Centric_Response2,
    IS_NICHE_OR_BROAD_MARKET2: Is_Niche_or_Broad_Market_Response2,
    IS_PRODUCT_OR_PLATFORM2: Is_Product_or_Platform_Response2,
    AI_STARTUP_TYPE3: AI_startup_type_Response3,
    DEVELOPER_OR_INTEGRATOR: Developer_or_Integrator_Response,
    AI_NATIVE_OR_AUGMENTED: AI_Native_or_Augmented_Response,
    AUTOMATION_DEPTH: Automation_Depth_Response
}


class LabelTask:
    def __init__(self, prompts: list[str]):
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=api_key
        )
        self.prompts = prompts
        
    def analyze_content(self, content: str, prompt: str) -> Dict:
        try:
            parser = JsonOutputParser(pydantic_object=map_response[prompt])
            prompt = PromptTemplate(
                template=prompt,
                input_variables=["content"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            chain = (prompt | self.llm | parser)
            result = chain.invoke({"content": content})
            return result
            
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            # Create a dictionary directly for error case
            return {
                "company_type": -1,
                "reasoning": "Error in analysis"
            }

    def process_xls(self, xls_path: str, output_path: str)->List[Dict]:
        """
        Process CSV file and analyze each article's content with flexible field mapping.
        
        Args:
            xls_path: Path to the xls file
            output_path: Path to save the JSON output and csv file
        """
        # Read xls
        df = pd.read_excel(xls_path)
        
        # Pre-process articles to include formatted content
        articles = df.to_dict('records')
        for prompt in tqdm(self.prompts):
            articles = self.run_analysis(articles, prompt)

        self.save_to_csv(articles, output_path)
        return articles
    
    def run_analysis(self, articles: List[Dict], prompt: str):
        # processed_articles = self._format_articles_content(articles)
        ans = asyncio.run(llm_articles(articles, self.llm, prompt))
        return ans

    def save_to_csv(self, ans: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ans, f, ensure_ascii=False, indent=4)
        df = pd.DataFrame(ans)
        df.to_csv(output_path.replace('.json', '.csv'), index=False)
        print(f"Analysis complete. Results saved to {output_path}")
    
    def _format_articles_content(self, articles: List[Dict]) -> List[Dict]:
        """
        Format the content of each article by combining specified fields and adding date information.
        """
        processed_articles = []
        for article in articles:
            processed_article = article.copy()
            content = article['investee_company_long_business_d']
            processed_article['content'] = content
            processed_articles.append(processed_article)
        return processed_articles


async def llm_articles(articles: List[Dict], model, prompt, batch_size: int = 10):
    """
    Process a list of articles using an LLM model with flexible schema handling.
    
    Args:
        articles: List of article dictionaries with 'formatted_content' field
        model: LLM model to use for processing
        batch_size: Number of articles to process in each batch
        
    Returns:
        List of processed articles with analysis
    """
    # Configure the output parser
    parser = JsonOutputParser(pydantic_object=map_response[prompt])
    
    prompt = PromptTemplate(
        template=prompt,
        input_variables=["content"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | model | parser
    
    
    async def batch_process_text(texts: List):
        return await chain.abatch(texts)
        
    # Create a list to store all answers
    ans = []
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        try:
            # Prepare batch inputs using the formatted content
            batch_inputs = [{"content": article['investee_company_long_business_d']} for article in batch]
            
            # Process the batch
            batch_responses = await batch_process_text(batch_inputs)
            
            # Process responses and add metadata
            for j, response in enumerate(batch_responses):
                article = batch[j]
                print(f"\n=== Article {i + j + 1}/{len(articles)} ===")
                
                # Create result entry with analysis and preserve all original article fields
                result_entry = {**response, **article}
                ans.append(result_entry)
                
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            continue

    return ans

if __name__ == "__main__":
    # Replace with your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    print(map_response[CORE_OR_GENERAL_AI_APPLICATION])
    xls_path = "/Users/wbik/Downloads/label-task/250425aistartup_sdc_crunch_des_smallsample.xls"
    # prompt_name = "base_prompt"
    # iea = LabelTask(base_prompt)
    # prompt_name = "modified_prompt"
    # iea = LabelTask(modified_prompt)
    prompt_name = "additional_prompt"
    iea = LabelTask(additional_prompt)
    output_path = f"small_sample_label_{prompt_name}.csv"
    ans = iea.process_xls(xls_path, output_path)