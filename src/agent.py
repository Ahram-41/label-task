from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from prompt import apply_prompt_template, agent_prompt_mapping
from prompt.prompts_control import *
from tools import tavily_tool
from langchain_core.output_parsers import JsonOutputParser
from basedata_control import *
from langchain_core.prompts import PromptTemplate
import pandas as pd
import json
from dotenv import load_dotenv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

load_dotenv()
map_response = {
    FOUNDERS_BACKGROUND: Founders_Background_Response,
    COMPANY_EXECUTIVES: Company_Executives_Response,
    AI_PRODUCTS: AI_Products_Response,
    AI_TECHNOLOGY: AI_Technology_Response,
    AI_PARTNER: AI_Partner_Response,
    IPO_MNA: IPO_MnA_Response
}
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=api_key,
    # tools=[{"type": "web_search"}]
)

def sanitize_filename(filename):
    """Remove or replace characters that are not allowed in filenames"""
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove any trailing dots or spaces
    sanitized = sanitized.strip('. ')
    # Limit length to avoid issues with long filenames
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized

def check_for_api_errors(messages, company_name):
    """Check for API errors in tool messages and raise appropriate exceptions"""
    api_errors = ["432 Client Error", "429 Client Error"]
    
    for message in messages:
        if message.type == "tool":
            content_str = str(message.content)
            for error in api_errors:
                if error in content_str:
                    error_type = "rate_limit" if "429" in error else "client_error"
                    raise Exception(f"{error} detected for {company_name}: {content_str}", error_type)

def process_single_company(company, task, query, research_agent, parser):
    """Process a single company and return the result"""
    kwargs = {
        "company_name": company["investee_company_name"],
        "description": company["investee_company_long_business_d"],
        "format_instructions": parser.get_format_instructions()
    }
    prompt = PromptTemplate(
        template=query,
        input_variables=["format_instructions"],
    ).format(**kwargs)
    print(f"Processing: {company['investee_company_name']}")

    response = research_agent.invoke({"messages": [("human", prompt)]})
    token_usage = response['messages'][-1].response_metadata['token_usage']
    completion_tokens = token_usage['completion_tokens']
    prompt_tokens = token_usage['prompt_tokens']
    
    # Check for API errors in tool messages
    check_for_api_errors(response['messages'], company['investee_company_name'])
    
    # Convert messages to a serializable format
    serializable_messages = []
    for msg in response['messages']:
        message_dict = {
            'type': msg.type,
            'content': msg.content,
        }
        if hasattr(msg, 'response_metadata'):
            message_dict['response_metadata'] = msg.response_metadata
        if hasattr(msg, 'artifact'):
            message_dict['artifact'] = msg.artifact
        serializable_messages.append(message_dict)
        
    # Save the serialized messages to a json file
    sanitized_company_name = sanitize_filename(company["investee_company_name"])
    with open(f"json_messages/response_{sanitized_company_name}_{task}_tavily.json", "w") as f:
        json.dump(serializable_messages, f, indent=2)

    tool_calls = []
    for message in response['messages']:
        if message.type == "tool":
            tool_calls.append((message.artifact))
    
    return {**company, "founders": response['structured_response'], "tool_calls": tool_calls, "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens}

def test_csv(task, start_index: int, batch_size: int = 10):
    query = agent_prompt_mapping[task]
    research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=lambda state: apply_prompt_template(task, map_response[query],state),
    response_format=map_response[query]
)
    parser = JsonOutputParser(pydantic_object=map_response[query])
    # df = pd.read_csv("others/distinct_selected_companies.csv")
    df = pd.read_excel("/Users/wbik/Downloads/label-task/250425aistartup_sdc_crunch_des_fullsample.xls")
    companies = df.to_dict('records')[start_index:]
    
    # Create json_messages directory if it doesn't exist
    os.makedirs("json_messages", exist_ok=True)
    
    total_companies = len(companies)
    print(f"Total companies to process: {total_companies}")
    print(f"Using {batch_size} parallel workers per batch")
    
    # Process companies in batches
    for batch_start in range(0, total_companies, batch_size):
        batch_end = min(batch_start + batch_size, total_companies)
        batch_companies = companies[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: companies {batch_start + 1} to {batch_end}")
        
        result = []
        api_error_occurred = 0
        
        # Use ThreadPoolExecutor for parallel processing within each batch
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all company processing tasks
            future_to_company = {
                executor.submit(process_single_company, company, task, query, research_agent, parser): company 
                for company in batch_companies
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_company):
                try:
                    company_result = future.result()
                    result.append(company_result)
                    print(f"Completed: {company_result['investee_company_name']}")
                except Exception as exc:
                    company = future_to_company[future]
                    
                    # Handle different types of API errors differently
                    if "429 Client Error" in str(exc):
                        # Rate limit - wait and continue processing
                        print(f"Rate limit hit for {company['investee_company_name']}: {exc}")
                        print("Rate limit hit - waiting 60 seconds before continuing...")
                        api_error_occurred = 2
                        # for remaining_future in future_to_company:
                        #     if not remaining_future.done():
                        #         remaining_future.cancel()
                        # break
                        
                    elif "432 Client Error" in str(exc):
                        # Client error - stop all processing
                        print(f"Client error detected for {company['investee_company_name']}: {exc}")
                        api_error_occurred = 1
                        # # Cancel remaining futures to stop processing
                        # for remaining_future in future_to_company:
                        #     if not remaining_future.done():
                        #         remaining_future.cancel()
                        # break
                        
                    else:
                        print(f"Company {company['investee_company_name']} generated an exception: {exc}")
                        #save in an error json file
                        with open(f"error_{task}.json", "a") as f:
                            #company name, error message, error type
                            error_data = {
                                "company_name": company['investee_company_name'],
                                "error_message": str(exc),
                                "error_type": type(exc).__name__
                            }
                            json.dump(error_data, f)
                        api_error_occurred = 1
                    # Cancel remaining futures to stop processing
                    for remaining_future in future_to_company:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    break
        
        # Save successful results from current batch if any
        if result:
            batch_df = pd.DataFrame(result)
            
            # Write header only for the first batch
            output_file = f"full_{task}.csv"
            write_header = not os.path.exists(output_file)
            mode = 'a' if os.path.exists(output_file) else 'w'
            
            batch_df.to_csv(output_file, index=False, mode=mode, header=write_header, encoding='utf-8-sig')
            print(f"Saved {len(result)} successful results from batch {batch_start//batch_size + 1}")
        
        # If API error occurred, stop processing
        if api_error_occurred != 0:
            print("Stopping processing due to 432 Client Error from Tavily API")
            print(f"Successfully processed {len(result)} companies in the current batch before error")
            return api_error_occurred, batch_end
        
        print(f"Batch {batch_start//batch_size + 1} completed and saved to full_{task}.csv")
    return api_error_occurred, batch_end

if __name__ == "__main__":
    start_index=2084
    while True:
        api_error, batch_end = test_csv("executive", start_index=start_index, batch_size=8)
        if api_error == 1:
            print("Stopping processing due to 432 Client Error from Tavily API")
            break
        elif api_error == 2:
            print("Rate limit hit, waiting for 120 seconds before continuing...")
            time.sleep(120)
            start_index += batch_end
        else:
            print("Processing completed successfully")
            break

