from typing import Annotated, TypedDict, Optional
import requests
import time
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Paper(TypedDict):
    id: str
    title: str
    year: int
    citations: int
    url: str

def search_research_papers(
    topic: Annotated[str, "The topic of the research paper"],
    year: Annotated[Optional[int], "Publication year"] = None,
    year_condition: Annotated[Optional[str], "One of 'in', 'before', 'after'"] = None,
    min_citations: Annotated[Optional[int], "Minimum number of citations"] = None
) -> Annotated[list[Paper], "A list of research papers matching the criteria"]:
    """
    Search for research papers using the Semantic Scholar API.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": topic,
        "fields": "paperId,title,year,citationCount",
        "limit": 10
    }

    if year is not None and year_condition == "in":
        params["year"] = str(year)
    elif year is not None and year_condition in ["before", "after"]:
        params["year"] = f"-{year - 1}" if year_condition == "before" else f"{year + 1}-"
    if min_citations is not None:
        params["minCitationCount"] = min_citations

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                    logger.info(
                        f"Rate limit exceeded (429). Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Max retries reached. Rate limit exceeded (429).")
                    raise Exception("Rate limit exceeded (429) after maximum retries")
            elif response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

            data = response.json().get("data", [])
            papers: list[Paper] = [
                {
                    "id": paper["paperId"],
                    "title": paper["title"] or "No title available",
                    "year": paper["year"] or 0,
                    "citations": paper["citationCount"] or 0,
                    "url": f"https://www.semanticscholar.org/paper/{paper['paperId']}"
                }
                for paper in data
            ]




            return papers

        except requests.RequestException as e:
            logger.error(f"Error querying Semantic Scholar API: {e}")
            return []