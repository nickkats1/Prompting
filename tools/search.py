from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import tool


@tool("web-search")
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    ddg = DuckDuckGoSearchResults(num_results=5)
    return ddg.invoke(query)

