from langchain_community.tools import TavilySearchResults

def initialize_tavily_tool():
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
    )
