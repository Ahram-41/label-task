import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from .logger import create_logged_tool
import os 
TAVILY_MAX_RESULTS = 5

# Get Tavily API key from environment variables
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable not set")

logger = logging.getLogger(__name__)

# Initialize Tavily search tool with logging
LoggedTavilySearch = create_logged_tool(TavilySearchResults)
tavily_tool = LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS, tavily_api_key=tavily_api_key)
