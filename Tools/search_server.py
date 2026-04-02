import urllib.request
import urllib.parse
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search")


def _ddg_search(query: str, max_results: int = 3) -> list:
    params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
    url = f"https://api.duckduckgo.com/?{params}"
    try:
        with urllib.request.urlopen(url, timeout=6) as r:
            data = json.loads(r.read().decode())
        results = []
        if data.get("AbstractText"):
            results.append({"title": data.get("Heading", query), "content": data["AbstractText"]})
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({"title": topic.get("Text", "")[:60], "content": topic["Text"]})
            if len(results) >= max_results:
                break
        return results[:max_results]
    except Exception as e:
        return [{"title": "Error", "content": str(e)}]


@mcp.tool()
def search_web(query: str) -> str:
    results = _ddg_search(query)
    if not results:
        return f"No results found for: '{query}'"
    return "\n\n".join([f"[{i+1}] {r['title']}\n    {r['content']}" for i, r in enumerate(results)])


@mcp.tool()
def search_news(query: str) -> str:
    results = _ddg_search(f"{query} news", max_results=3)
    if not results:
        return f"No news found for: '{query}'"
    return "\n\n".join([f"[{i+1}] {r['title']}\n    {r['content']}" for i, r in enumerate(results)])


if __name__ == "__main__":
    mcp.run(transport="stdio")
