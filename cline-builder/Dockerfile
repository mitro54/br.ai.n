# Multi-Agent Cline Builder
# Node 24 + Python 3 for running @cline/cli and distillation scripts
FROM node:24-slim

# Install Python 3 and system dependencies (including fast search utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    jq \
    procps \
    git \
    tree \
    ripgrep \
    fd-find \
    && ln -sf $(which fdfind || echo /usr/bin/fdfind) /usr/local/bin/fd \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (use --break-system-packages for Debian 12+)
RUN pip3 install --break-system-packages httpx

# Install cline cli globally
RUN npm install -g cline

# Set up application directory
WORKDIR /app

# Copy pipeline scripts, prompts, MCP server, and CLI tools
COPY entrypoint.sh /app/entrypoint.sh
COPY distill.py /app/distill.py
COPY mcp_server.py /app/mcp_server.py
COPY prompts/ /app/prompts/
COPY tools/ /app/tools/
COPY tools/web_search.py /usr/local/bin/web-search
COPY tools/fetch_page.py /usr/local/bin/fetch-page
COPY tools/kb_search.py /usr/local/bin/kb-search
RUN chmod +x /app/entrypoint.sh /app/mcp_server.py /usr/local/bin/web-search /usr/local/bin/fetch-page /usr/local/bin/kb-search

# Workspace is mounted at runtime
WORKDIR /workspace

ENTRYPOINT ["/app/entrypoint.sh"]
