#!/usr/bin/env python3
"""
SearXNG-Exclusive Web Search CLI Tool for Cline Builder
Queries the private SearXNG instance on the local Docker network.
Outputs clean markdown summaries of top search results.
"""
import os
import sys
import urllib.parse

import httpx

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")

def main():
    if len(sys.argv) < 2:
        print("Usage: web-search <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Error: Empty query")
        sys.exit(1)

    # Try configured SearXNG URL, then host.docker.internal fallback if running with host mapping
    candidate_urls = [
        SEARXNG_URL,
        "http://host.docker.internal:8080"
    ]
    # Remove duplicates
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
                    last_error = f"SearXNG returned HTTP {resp.status_code}"
            except Exception as e:  # noqa: BLE001
                last_error = str(e)

    if not data:
        print(f"SearXNG search unavailable ({last_error}). Ensure SearXNG container is active.")
        sys.exit(1)

    results = data.get("results", [])[:5]
    if not results:
        print(f"No SearXNG results found for: '{query}'")
        return

    print(f"### SearXNG Search Results for: `{query}`\n")
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        link = r.get("url", "")
        content = r.get("content", "").strip()
        print(f"**{i}. [{title}]({link})**")
        if content:
            print(f"> {content}\n")
        else:
            print()

if __name__ == "__main__":
    main()
