#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for br.ai.n / Cline Builder
Exposes native tools for SearXNG Web Search, Web Page Fetching, and Local Knowledge Base search.
Runs over standard stdio JSON-RPC 2.0.
"""
import json
import os
import sys
import urllib.parse

# Ensure the script's directory is always on the module search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import httpx

# Re-use logic from tool modules
from tools.kb_search import search_kb

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")


def execute_web_search(query: str) -> str:
    """Query SearXNG for web search results."""
    candidate_urls = [SEARXNG_URL, "http://host.docker.internal:8080"]
    candidate_urls = list(dict.fromkeys(candidate_urls))

    data = None
    last_error = None

    with httpx.Client(timeout=10.0) as client:
        for base_url in candidate_urls:
            try:
                url = f"{base_url}/search?q={urllib.parse.quote(query)}&format=json"
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    break
                else:
                    last_error = f"HTTP {resp.status_code}"
            except Exception as e:  # noqa: BLE001
                last_error = str(e)

    if not data:
        return f"SearXNG search unavailable ({last_error}). Ensure SearXNG container is active."

    results = data.get("results", [])[:5]
    if not results:
        return f"No results found for: '{query}'"

    lines = [f"### SearXNG Search Results for: `{query}`\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        link = r.get("url", "")
        content = r.get("content", "").strip()
        lines.append(f"**{i}. [{title}]({link})**")
        if content:
            lines.append(f"> {content}\n")
        else:
            lines.append("")
    return "\n".join(lines)


def execute_fetch_page(url: str, max_chars: int = 15000) -> str:
    """Fetch URL and return clean text/markdown."""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return f"Failed to fetch {url}: HTTP {resp.status_code}"

            html = resp.text
            # Basic tag stripping and cleaning
            import re
            text = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... [Truncated at {max_chars} characters]"
            return f"### Content from `{url}`\n\n{text}"
    except Exception as e:  # noqa: BLE001
        return f"Error fetching {url}: {e}"


# Tool definitions for MCP
TOOLS = [
    {
        "name": "searxng_web_search",
        "description": "Perform a private live web search using local SearXNG to look up framework documentation, APIs, and code examples.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query keywords"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_web_page",
        "description": "Fetch documentation and web page contents from a specific URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the local project knowledge base (.knowledge_base/) for architectural patterns, component standards, and conventions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or topic to look up in the project knowledge base"
                }
            },
            "required": ["query"]
        }
    }
]


def handle_request(req: dict) -> dict | None:
    """Handle incoming JSON-RPC 2.0 MCP request."""
    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "brain-mcp-server",
                    "version": "1.0.0"
                }
            }
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        result_text = ""
        if tool_name == "searxng_web_search":
            query = args.get("query", "")
            result_text = execute_web_search(query)
        elif tool_name == "fetch_web_page":
            url = args.get("url", "")
            result_text = execute_fetch_page(url)
        elif tool_name == "search_knowledge_base":
            query = args.get("query", "")
            result_text = search_kb(query)
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }
        }
    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {}
        }
    else:
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not supported: {method}"
                }
            }
        return None


def main():
    """Main stdio loop for MCP server."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as e:  # noqa: BLE001
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {e}"
                }
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
