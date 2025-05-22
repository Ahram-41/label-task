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

load_dotenv()
map_response = {
    FOUNDERS_BACKGROUND: Founders_Background_Response,
    COMPANY_EXECUTIVES: Company_Executives_Response,
    AI_PRODUCTS: AI_Products_Response,
    AI_TECHNOLOGY: AI_Technology_Response
}
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=api_key,
    # tools=[{"type": "web_search"}]
)



def test_one():
    kwargs = {
        # "company_name": "Advanced Recognition Technologies, Inc.",
        # "description": "Advanced Recognition Technologies, Inc. provides handwriting-recognition and voice-activation software products for mass-communication devices. The company's advanced recognizers, capable of running on low-level processors addresses the market for pen-based computers and palm-held processing devices. Handwriting Recognition Software",
        "company_name": "Instrumental Inc",
        "description": "Instrumental Inc is a United States-based company, which provides manufacturing platform that transforms data captured on assembly lines into insights. Its instrumental captures data from the assembly line. The Company's artificial intelligence (AI) algorithms processes data in real-time. The Company serves industries, including consumer electronics, automotive, medical and aerospace. UNIX-based software performance monitoring capacity planning architecting",
        "format_instructions": parser.get_format_instructions()
    }
    query = PromptTemplate(
            template=query,
            input_variables=["format_instructions"],
        ).format(**kwargs)
    print(query)
    for step in research_agent.stream(
        {"messages": [("human", query)]}, stream_mode="updates"
    ):
        print(step)
    print("end")
    # response = research_agent.invoke({"messages": [("human", query)]})
    # print(response)
    # ans =llm.invoke(system_prompt)
    # print(ans)
def test_csv(task):
    query = agent_prompt_mapping[task]
    research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=lambda state: apply_prompt_template(task, map_response[query],state),
    response_format=map_response[query]
)
    parser = JsonOutputParser(pydantic_object=map_response[query])
    df = pd.read_csv("distinct_selected_companies.csv")
    companies = df.to_dict('records')
    print(len(companies[:21]))
    result = []
    for company in companies[:20]:
        kwargs = {
            "company_name": company["investee_company_name"],
            "description": company["investee_company_long_business_d"],
            "format_instructions": parser.get_format_instructions()
        }
        prompt = PromptTemplate(
            template=query,
            input_variables=["format_instructions"],
        ).format(**kwargs)
        print(prompt)

        response = research_agent.invoke({"messages": [("human", prompt)]})
        token_usage = response['messages'][-1].response_metadata['token_usage']
        completion_tokens = token_usage['completion_tokens']
        prompt_tokens = token_usage['prompt_tokens']
        
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
        with open(f"json_messages/response_{company['investee_company_name']}_{task}_tavily.json", "w") as f:
            json.dump(serializable_messages, f, indent=2)

        tool_calls = []
        for message in response['messages']:
            if message.type == "tool":
                tool_calls.append((message.artifact))
        result.append({**company, "founders": response['structured_response'], "tool_calls": tool_calls, "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens})
    df = pd.DataFrame(result)
    df.to_csv(f"{task}.csv", index=False, mode='w', header=True)


def test_openai_search():
    df = pd.read_csv("distinct_selected_companies.csv")
    companies = df.to_dict('records')
    print(len(companies[:21]))
    result = []
    for company in companies[:2]:
        kwargs = {
            "company_name": company["investee_company_name"],
            "description": company["investee_company_long_business_d"],
            # "format_instructions": parser.get_format_instructions()
        }
        
        query1 = """Who are the founders of {company_name}?

your searching SEO should includes company name and "Founder"

COMPANY NAME: {company_name}
DESCRIPTION: {description}
"""
        prompt = PromptTemplate(
            template=query1,
            # input_variables=["format_instructions"],
        ).format(**kwargs)
        print(prompt)

        response = llm.invoke(prompt)
        token_usage = response.usage_metadata
        completion_tokens = token_usage['output_tokens']
        prompt_tokens = token_usage['input_tokens']
        tool_calls = []
        for content in response.content:
            tool_calls.extend(content['annotations'])

        query2="""
        Based on the FOUNDER team members and the company name, search the education background and ipo, M&A information of each founder. your may search the information from web like LinkedIn, Crunchbase, etc. if there's no Founder team member names, return "No information found"
        {company_name}
        {description}
        {format_instructions}
        """

        temp = PromptTemplate(
            template=query2,
            input_variables=["content", "description"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = (temp | llm | parser)
        try:
            ans = chain.invoke({"company_name": company["investee_company_name"], "description": content})
        except Exception as e:
            ans = "No information found"
    
        result.append({**company, "founders": ans, "tool_calls": tool_calls,"content": content, "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens})
    df = pd.DataFrame(result)
    df.to_csv("founders_background_openai.csv", index=False, mode='w', header=False)

if __name__ == "__main__":
    test_csv("technology")
