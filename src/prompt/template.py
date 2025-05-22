import os
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel
# from mappings import agent_prompt_mapping, map_response
def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    # Escape curly braces using backslash
    template = template.replace("{", "{{").replace("}", "}}")
    template = re.sub(r"<<([^>>]+)>>", r"{\1}", template)
    return template


def apply_prompt_template(prompt_name: str, response_format: BaseModel, state: AgentState) -> list:
    parser = JsonOutputParser(pydantic_object=response_format)
    system_prompt = PromptTemplate(
        template=get_prompt_template(prompt_name),
        # input_variables=["format_instructions"],
    ).format(**state)
    return [{"role": "system", "content": system_prompt}] + state["messages"]
