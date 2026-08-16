#!/usr/bin/env python3
"""
Safe Web Documentation / Page Fetcher for Cline Builder
Fetches a webpage, strips scripts and styles, and prints sanitized readable text.
Limits character count to 6000 characters to protect LLM context windows.
"""
import sys
import re
import httpx

MAX_CHARS = 6000

def sanitize_html(html: str) -> str:
    # Remove script and style tags
    cleaned = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<style.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<noscript.*?</noscript>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<header.*?</header>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<footer.*?</footer>>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<nav.*?</nav>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert headings and paragraphs
    cleaned = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n### \1\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<pre[^>]*>(.*?)</pre>', r'\n```\n\1\n```\n', cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Strip remaining HTML tags
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    
    # Normalize whitespace and HTML entities
    cleaned = cleaned.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: fetch-page <url>")
        sys.exit(1)

    url = sys.argv[1].strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"Failed to fetch page (HTTP {resp.status_code})")
                sys.exit(1)

            text = sanitize_html(resp.text)
            if not text:
                print("Page returned empty content.")
                return

            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "\n\n... [Content truncated for context safety]"

            print(f"### Documentation from `{url}`:\n")
            print(text)
    except Exception as e:
        print(f"Error fetching page: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
