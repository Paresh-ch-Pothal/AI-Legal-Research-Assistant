"""
web_search.py
─────────────
Lightweight DuckDuckGo web-search helper (via LangChain) used to pull in
live, open-web context alongside the locally indexed case documents.

Requires:
    pip install -U langchain-community duckduckgo-search

Usage (from app.py):
    from web_search import search_web, format_results_as_markdown

    results = search_web("Section 498A IPC recent judgments", max_results=5)
    markdown_block = format_results_as_markdown(results)
"""

from typing import List, Dict


def search_web(query: str, max_results: int = 5, region: str = "in-en") -> List[Dict]:
    """
    Run a DuckDuckGo search through LangChain and return a normalised list
    of result dicts: [{"title": ..., "link": ..., "snippet": ...}, ...]

    This never raises — on failure (no internet, DDG rate-limiting, missing
    dependency, etc.) it logs the error and returns an empty list so the
    chat flow is never interrupted.
    """
    if not query or not query.strip():
        return []

    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

        wrapper = DuckDuckGoSearchAPIWrapper(region=region, max_results=max_results)
        raw_results = wrapper.results(query, max_results=max_results)

        normalised = []
        for r in raw_results[:max_results]:
            normalised.append({
                "title":   r.get("title", "Untitled Result"),
                "link":    r.get("link", ""),
                "snippet": r.get("snippet", ""),
            })
        return normalised

    except Exception as e:
        # Network issues, DuckDuckGo throttling, or a missing dependency
        # should never crash the chat — degrade quietly.
        print(f"[web_search] DuckDuckGo search failed: {e}")
        return []


def format_results_as_markdown(results: List[Dict]) -> str:
    """
    Turn a list of search results into a short markdown block that can be
    appended directly to an assistant's answer.
    """
    if not results:
        return ""

    lines = ["", "---", "### 🌐 Related Web Results (DuckDuckGo)"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled Result")
        link = r.get("link", "")
        snippet = r.get("snippet", "")
        if link:
            lines.append(f"{i}. **[{title}]({link})**  \n   {snippet}")
        else:
            lines.append(f"{i}. **{title}**  \n   {snippet}")
    return "\n".join(lines)