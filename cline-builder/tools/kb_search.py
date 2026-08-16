#!/usr/bin/env python3
"""
Knowledge Base Search CLI Tool for Cline Builder
Searches markdown documentation in /workspace/.knowledge_base/
Outputs clean summaries and relevant snippets.
"""
import glob
import os
import re
import sys

KB_DIR = os.environ.get("KB_DIR", "/workspace/.knowledge_base")


def search_kb(query: str, kb_dir: str = KB_DIR, max_results: int = 5) -> str:
    """Search knowledge base files for relevant keywords and snippets."""
    if not os.path.exists(kb_dir):
        return f"Knowledge base directory ({kb_dir}) not found."

    raw_words = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    stop_words = {"the", "a", "an", "is", "it", "to", "and", "or", "of", "in", "on", "for", "with", "this", "that"}
    keywords = [w for w in raw_words if len(w) >= 3 and w not in stop_words]

    if not keywords:
        return "Please provide specific search keywords."

    md_files = glob.glob(f"{kb_dir}/**/*.md", recursive=True)
    if not md_files:
        return f"No markdown files found in {kb_dir}."

    scored = []
    for path in md_files:
        rel_path = os.path.relpath(path, kb_dir)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue

        score = 0
        lower_rel = rel_path.lower()
        lower_content = content.lower()

        for kw in keywords:
            if kw in lower_rel:
                score += 15
            occurrences = lower_content.count(kw)
            score += min(occurrences * 2, 20)

        if score > 0:
            scored.append((score, rel_path, content))

    scored.sort(key=lambda x: -x[0])
    if not scored:
        return f"No matching knowledge base documentation found for: '{query}'"

    output = [f"### 📖 Knowledge Base Search Results for: `{query}`\n"]
    for i, (score, rel_path, content) in enumerate(scored[:max_results], 1):
        output.append(f"**{i}. {rel_path}**")
        # Extract first 500 characters or key paragraph
        preview = content.strip()[:800]
        output.append(f"```markdown\n{preview}\n```\n")

    return "\n".join(output)


def main():
    if len(sys.argv) < 2:
        print("Usage: kb-search <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Error: Empty query")
        sys.exit(1)

    result = search_kb(query)
    print(result)


if __name__ == "__main__":
    main()
