#!/usr/bin/env python3
"""
Multi-Pass Context Distillation Engine

Reads a conversation JSON file, runs it through 4 expert LLM passes
(Architect → Engineer → Test Engineer → Safety Inspector), and writes a combined
.clinerules file for the Cline CLI agent.

Handles conversation chunking when content exceeds the context window.
Manages Ollama model loading/unloading between passes for VRAM safety.
Isolates context between passes using Markdown boundaries.
"""

import json
import os
import time
import httpx
import threading

# --- Configuration ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://host.docker.internal:8000")
CONFIG_PATH = os.environ.get("AGENT_CONFIG_PATH", "/app/agent_config.json")
CONVERSATION_PATH = os.environ.get("CONVERSATION_FILE", "/workspace/.cline_context/conversation.json")
OUTPUT_PATH = os.environ.get("CLINERULES_PATH", "/workspace/.clinerules")
STATUS_PATH = os.environ.get("DISTILL_STATUS_PATH", "/workspace/.cline_context/distill_status")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "unnamed_project")
CONTEXT_WINDOW = int(os.environ.get("EXPERT_CTX", "16384"))

# Rough approximation: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4
# Reserve tokens for system prompt + response
RESERVED_TOKENS = 2048
CHUNK_OVERLAP_TOKENS = 200

# Keep the chunk size small to prevent CPU ingestion stalls
TARGET_CHUNK_SIZE = 2048


def load_config() -> dict:
    """Load the agent configuration file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_prompt(prompt_val: str, config_path: str = CONFIG_PATH) -> str:
    """
    Resolve a prompt entry. If prompt_val is a path to a file (e.g. 'prompts/architect.md'),
    load and return the file content. Otherwise, return the string as is.
    """
    if not prompt_val or not isinstance(prompt_val, str):
        return ""

    if "\n" not in prompt_val and (prompt_val.endswith(".md") or prompt_val.endswith(".txt") or "/" in prompt_val):
        candidates = [
            prompt_val,
            os.path.join(os.path.dirname(config_path), prompt_val),
            os.path.join("/app", prompt_val),
            os.path.join(os.path.dirname(__file__), prompt_val),
            os.path.join(os.getcwd(), prompt_val),
            os.path.join(os.getcwd(), "cline-builder", prompt_val),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            return content
                except Exception as e:
                    print(f"  ⚠ Failed to read prompt file {candidate}: {e}")
    return prompt_val


def load_conversation() -> list:
    """Load the conversation messages from the JSON file."""
    with open(CONVERSATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def conversation_to_text(messages: list) -> str:
    """Flatten conversation messages into a readable text block."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if content.strip():
            parts.append(f"[{role}]\n{content}")
    return "\n\n---\n\n".join(parts)


def _resolve_model_config(model_entry, default_host: str = None) -> dict:
    """
    Resolve a model entry from config into a normalized dict.
    Supports both legacy string format and new object format.
    """
    if default_host is None:
        default_host = OLLAMA_HOST
    if isinstance(model_entry, str):
        return {"model": model_entry, "provider": "ollama", "base_url": default_host}
    return {
        "model": model_entry.get("model", ""),
        "provider": model_entry.get("provider", "ollama"),
        "base_url": model_entry.get("base_url", default_host),
        "args": model_entry.get("args", []),
    }


def chunk_text(text: str, max_tokens: int) -> list[str]:
    """
    Split text into chunks that fit within the token budget.
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunks.append(text[start:])
            break

        break_point = text.rfind("\n\n", start + max_chars // 2, end)
        if break_point == -1:
            break_point = text.rfind("\n", start + max_chars // 2, end)
        if break_point == -1:
            break_point = text.rfind(". ", start + max_chars // 2, end)
        if break_point != -1:
            end = break_point + 1

        chunks.append(text[start:end])

        new_start = end - overlap_chars
        if new_start <= start:
            new_start = start + 1
        start = new_start

    return chunks


def unload_model(client: httpx.Client, model_config: dict):
    """
    Unload a model from VRAM using the appropriate provider method.
    For Ollama: direct API call. For others: calls the orchestrator management API.
    """
    model_name = model_config.get("model", "") if isinstance(model_config, dict) else model_config
    provider = model_config.get("provider", "ollama") if isinstance(model_config, dict) else "ollama"
    base_url = model_config.get("base_url", OLLAMA_HOST) if isinstance(model_config, dict) else OLLAMA_HOST

    try:
        if provider == "ollama":
            client.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10.0
            )
            print(f"  ↳ Unloaded model (Ollama): {model_name}")
        else:
            # Non-Ollama: call the orchestrator's management API
            client.post(
                f"{ORCHESTRATOR_URL}/internal/model/unload",
                json=model_config if isinstance(model_config, dict) else {"model": model_config, "provider": "ollama"},
                timeout=30.0
            )
            print(f"  ↳ Unloaded model ({provider}): {model_name}")
        time.sleep(1)
    except Exception as e:
        print(f"  ⚠ Failed to unload {model_name}: {e}")


def call_llm(client: httpx.Client, model_config, system_prompt: str, user_content: str, prior_context: str = "") -> str:
    """
    Send a synchronous chat completion request to the configured provider.
    Uses generic extraction prompts for chunks to prevent template deadlocks.
    Accepts model_config as either a string (legacy Ollama) or a dict with provider info.
    """
    prior_tokens = len(prior_context) // CHARS_PER_TOKEN
    available_tokens = CONTEXT_WINDOW - RESERVED_TOKENS - prior_tokens

    if available_tokens < 1000:
        available_tokens = 1000

    chunk_limit = min(available_tokens, TARGET_CHUNK_SIZE)
    chunks = chunk_text(user_content, chunk_limit)

    print(f"  ↳ Input: {len(user_content) + len(prior_context)} chars total. (Ctx: {CONTEXT_WINDOW}, Chunk Limit: {chunk_limit})", flush=True)

    if len(chunks) == 1:
        full_input = ""
        if prior_context:
            full_input += f"### PREVIOUS ANALYSES\n{prior_context}\n\n---\n\n"
        full_input += f"### CURRENT TASK\n{user_content}"

        print(f"    ↳ Preparing Single-pass (Ingesting {len(full_input)} chars)...", flush=True)
        return _single_llm_call(client, model_config, system_prompt, full_input)

    print(f"    ↳ Processing into {len(chunks)} parts...", flush=True)
    partial_results = []

    # [FIX] A relaxed, generic system prompt for the chunks so it doesn't deadlock trying to fill out a template it doesn't have data for.
    relaxed_chunk_system_prompt = (
        "You are acting as an information extractor. Your final formatting goal will be defined later. "
        "For now, review the provided text chunk and extract ANY raw technical details, facts, or logical "
        "requirements that stand out. Output simple bullet points. Do not attempt to use formal templates."
    )

    for i, chunk in enumerate(chunks):
        part_label = f"Part {i + 1}/{len(chunks)}"

        chunk_prompt = ""
        if prior_context:
            chunk_prompt += f"### PREVIOUS ANALYSES\n{prior_context}\n\n---\n\n"

        chunk_prompt += (
            f"### CURRENT TASK\n"
            f"Extract key technical information from the following text chunk.\n\n"
            f"{chunk}\n\n"
            f"---\n"
            f"CHUNK IDENTIFIER: PART {i + 1} OF {len(chunks)}"
        )

        print(f"    ↳ Preparing {part_label} (Payload: {len(chunk_prompt)} chars)...", flush=True)
        # Use the relaxed prompt for the chunks
        result = _single_llm_call(client, model_config, relaxed_chunk_system_prompt, chunk_prompt, part_label)
        partial_results.append(result)

    print("    ↳ All parts finished. Starting Merge Pass...", flush=True)

    # Recursive merge if facts are too large
    merged_facts = "\n\n".join(partial_results)
    if len(merged_facts) > 60000:
        print(f"    ↳ Facts too large ({len(merged_facts)} chars). Running consolidation...", flush=True)
        buckets = [partial_results[i:i+5] for i in range(0, len(partial_results), 5)]
        consolidated = []
        for bi, bucket in enumerate(buckets):
            bucket_text = "\n\n".join(bucket)
            consolidation_prompt = (
                f"Consolidate these extracted facts into a concise summary. "
                f"Remove duplicates. Keep only unique technical requirements.\n\n{bucket_text}"
            )
            result = _single_llm_call(client, model_config,
                "You are a technical summarizer. Output concise bullet points only.",
                consolidation_prompt, f"Consolidation {bi+1}/{len(buckets)}")
            consolidated.append(result)
        partial_results = consolidated
        print(f"    ↳ Consolidated {len(buckets)} buckets into {len(consolidated)} summaries.", flush=True)

    # Now we apply strict system prompt to the merged bullets
    merge_prompt = (
        "You previously extracted technical details from a larger conversation in parts. "
        "Below are the raw extracted bullet points.\n\n"
        "Using ONLY these details (and the PREVIOUS ANALYSES if provided), write your final response. "
        "You MUST strictly adhere to your system prompt instructions and template formatting.\n\n"
    )
    for i, part in enumerate(partial_results):
        merge_prompt += f"#### EXTRACTED FACTS (PART {i + 1})\n{part}\n\n"

    # Use the REAL system prompt here
    return _single_llm_call(client, model_config, system_prompt, merge_prompt, "Merging Parts")


def _single_llm_call(client: httpx.Client, model_config, system_prompt: str, user_content: str, label: str = "Inference") -> str:
    """
    Execute a single LLM API call with streaming for live feedback.
    Supports both Ollama native and OpenAI-compatible streaming formats.

    Args:
        model_config: Either a string (model name, Ollama) or a dict with provider info.
    """
    # Normalize config
    if isinstance(model_config, str):
        cfg = {"model": model_config, "provider": "ollama", "base_url": OLLAMA_HOST}
    else:
        cfg = model_config

    model_name = cfg.get("model", "")
    provider = cfg.get("provider", "ollama")
    base_url = cfg.get("base_url", OLLAMA_HOST)
    is_ollama = provider == "ollama"

    if is_ollama:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": True,
            "options": {
                "num_ctx": CONTEXT_WINDOW,
                "temperature": 0.3,
            },
            "keep_alive": "3m"
        }
        url = f"{base_url}/api/chat"
    else:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": True,
            "temperature": 0.3,
        }
        url = f"{base_url}/v1/chat/completions"

    max_retries = 3
    for attempt in range(max_retries):
        first_token_received = threading.Event()

        def heartbeat():
            start_wait = time.time()
            while not first_token_received.is_set():
                time.sleep(5)
                if not first_token_received.is_set():
                    elapsed = int(time.time() - start_wait)
                    print(f"      ↳ [Waiting for LLM... {elapsed}s]", flush=True)

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            full_response = []
            print(f"    {label:15} [Generating...]\n    ↳ ", end="", flush=True)
            start_time = time.time()

            with httpx.Client() as stream_client:
                # Disable orchestrator scrubbing for distillation passes
                headers = {"X-No-Scrub": "true"}
                # Stability Protocol: 60s budget per part.
                try:
                    with stream_client.stream("POST", url, json=payload, headers=headers, timeout=60.0) as resp:
                        if resp.status_code == 503:
                            first_token_received.set()
                            print(f"\n  ⚠ Orchestrator is busy (503). Retrying in 10s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(10)
                            continue

                        if resp.status_code != 200:
                            first_token_received.set()
                            print("\n  ✗ LLM returned status", resp.status_code)
                            # Read error if possible
                            try:
                                err_body = resp.read().decode()
                                print(f"    Error: {err_body[:200]}")
                            except Exception:
                                pass
                            return f"[ERROR: LLM returned status {resp.status_code}]"

                        dot_count = 0
                        for line in resp.iter_lines():
                            # Stability Protocol: 60s hard budget per part (wall-clock time)
                            # This must run REGARDLESS of whether we received a token or a heartbeat.
                            if time.time() - start_time > 60:
                                if attempt < max_retries - 1:
                                    print(f"\n      ⚠ [STABILITY PROTOCOL] Analytical capacity exceeded (60s). Retrying part ({attempt+2}/{max_retries})...")
                                    first_token_received.set()
                                    raise httpx.ReadTimeout("Analytical capacity exceeded")
                                else:
                                    print(f"\n      ✗ [STABILITY PROTOCOL] Analytical capacity exceeded on FINAL ATTEMPT. Salvaging {len(full_response)} tokens.")
                                    first_token_received.set()
                                    return "".join(full_response)

                            if not line:
                                continue

                            if not first_token_received.is_set():
                                first_token_received.set()

                            if is_ollama:
                                chunk_data = json.loads(line)
                                token = chunk_data.get("message", {}).get("content", "")
                                done = chunk_data.get("done", False)
                            else:
                                if not line.startswith("data: "):
                                    # Handle orchestrator heartbeats and non-data SSE events
                                    continue
                                if line == "data: [DONE]":
                                    break
                                try:
                                    chunk_data = json.loads(line[6:])
                                    choices = chunk_data.get("choices", [{}])
                                    token = choices[0].get("delta", {}).get("content", "") if choices else ""
                                    done = choices[0].get("finish_reason") is not None if choices else False
                                except Exception:
                                    # Skip malformed chunks or internal proxy metadata
                                    continue

                            if token:
                                full_response.append(token)
                                dot_count += 1
                                if dot_count % 20 == 0:
                                    print(".", end="", flush=True)

                            if done:
                                break

                    elapsed = time.time() - start_time
                    print(f" ✓ ({elapsed:.1f}s)", flush=True)
                    return "".join(full_response)

                except httpx.ReadTimeout:
                    first_token_received.set()
                    if attempt < max_retries - 1:
                        print(f"\n      ⚠ [Stability Protocol] Analytical capacity exceeded (timeout). Retrying part ({attempt+2}/{max_retries})...")
                        continue
                    else:
                        print(f"\n      ✗ [Stability Protocol] Analytical capacity exceeded on FINAL ATTEMPT. Salvaging {len(full_response)} tokens.")
                        return "".join(full_response) if full_response else "[ERROR: ReadTimeout]"

        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            first_token_received.set()
            # If we get a 'Server disconnected' or 'Remote protocol error', it's often transient
            is_reset = "RemoteProtocolError" in str(type(e)) or "disconnected" in str(e).lower()

            if attempt < max_retries - 1:
                wait_time = 10 if is_reset else 5
                print(f"\n  ⚠ Network/Protocol error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            print(f"\n  ❌ LLM request failed after {max_retries} attempts: {e}")
            return f"[ERROR: {e}]"
        except Exception as e:
            first_token_received.set()
            print(f"\n  ❌ Unexpected error: {e}")
            return f"[ERROR: {e}]"

    return "[ERROR: Max retries exceeded]"


def update_status(status: str):
    """Write current status to a file in the workspace."""
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            f.write(status)
    except Exception as e:
        print(f"  ⚠ Failed to update status file: {e}")


# --- Stop words for KB keyword matching ---
STOP_WORDS = {"the","a","an","is","it","to","and","or","of","in","on","for",
              "with","this","that","from","be","as","at","by","we","do","if",
              "not","but","so","make","sure","lets","fix","then","can","will",
              "should","must","have","has","been","was","are","also","any",
              "all","just","get","set","use","new","add","now","our","its"}


def select_relevant_kb(kb_dir: str, instruction: str, max_chars: int = 100000) -> str:
    """Score and select only relevant KB files based on keyword matching."""
    import glob
    import re

    # 1. Extract keywords from instruction (3+ chars, not stop words)
    raw_words = re.findall(r'[a-zA-Z0-9_]+', instruction.lower())
    keywords = {w for w in raw_words if len(w) >= 3 and w not in STOP_WORDS}

    if not keywords:
        print("  📖 KB: No meaningful keywords found in instruction. Skipping KB.", flush=True)
        return ""

    print(f"  📖 KB Keywords: {', '.join(sorted(keywords)[:15])}", flush=True)

    # 2. Score each file
    scored_files = []
    for md_file in glob.glob(f"{kb_dir}/**/*.md", recursive=True):
        score = 0
        basename = os.path.basename(md_file).lower()

        # Filename match = high relevance
        for kw in keywords:
            if kw in basename:
                score += 10

        # Content peek match (first 500 chars only)
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                peek = f.read(500).lower()
            for kw in keywords:
                if kw in peek:
                    score += 5
        except Exception:
            continue

        scored_files.append((score, md_file))

    # 3. Sort by score (highest first)
    scored_files.sort(key=lambda x: -x[0])

    # 4. Inject full content for matches, filenames-only for the rest
    selected_content = []
    collateral_names = []
    total_chars = 0

    for score, path in scored_files:
        if score > 0 and total_chars < max_chars:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if total_chars + len(content) <= max_chars:
                    selected_content.append(f"### KB: {os.path.basename(path)}\n{content}")
                    total_chars += len(content)
                else:
                    remaining = max_chars - total_chars
                    selected_content.append(
                        f"### KB: {os.path.basename(path)} [TRUNCATED]\n{content[:remaining]}"
                    )
                    total_chars = max_chars
            except Exception:
                continue
        else:
            collateral_names.append(os.path.basename(path))

    result = "\n\n".join(selected_content)
    if collateral_names:
        result += "\n\n### Other KB Files (not loaded - request if needed):\n"
        result += ", ".join(collateral_names[:50])

    matched = len(selected_content)
    total = len(scored_files)
    print(f"  📖 KB Selection: {matched}/{total} files matched, {total_chars} chars injected (cap: {max_chars})", flush=True)
    return result


def detect_project_toolchain(project_dir: str) -> str:
    """Detect language, package manager, test runner, and formatter from project markers."""
    markers = {
        "pyproject.toml":   {"lang": "Python", "pkg": "poetry/pip", "fmt": "black/ruff", "test": "pytest"},
        "setup.py":         {"lang": "Python", "pkg": "pip", "fmt": "black", "test": "pytest"},
        "requirements.txt": {"lang": "Python", "pkg": "pip", "fmt": "black", "test": "pytest"},
        "Pipfile":          {"lang": "Python", "pkg": "pipenv", "fmt": "black", "test": "pytest"},
        "package.json":     {"lang": "JavaScript/TypeScript", "pkg": "npm/yarn", "fmt": "prettier", "test": "jest/vitest"},
        "tsconfig.json":    {"lang": "TypeScript", "pkg": "npm", "fmt": "prettier", "test": "jest"},
        "Cargo.toml":       {"lang": "Rust", "pkg": "cargo", "fmt": "rustfmt", "test": "cargo test"},
        "go.mod":           {"lang": "Go", "pkg": "go mod", "fmt": "gofmt", "test": "go test"},
        "CMakeLists.txt":   {"lang": "C/C++", "pkg": "cmake", "fmt": "clang-format", "test": "ctest/make test"},
        "Makefile":         {"lang": "C/C++", "pkg": "make", "fmt": "clang-format", "test": "make test"},
        "pom.xml":          {"lang": "Java", "pkg": "maven", "fmt": "google-java-format", "test": "mvn test"},
        "build.gradle":     {"lang": "Java/Kotlin", "pkg": "gradle", "fmt": "spotless", "test": "gradle test"},
    }

    detected = []
    seen_langs = set()
    for marker, info in markers.items():
        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in ["node_modules", ".git", "venv", ".venv", "__pycache__"]]
            if marker in files and info["lang"] not in seen_langs:
                detected.append(info)
                seen_langs.add(info["lang"])
                break

    if not detected:
        return ""

    result = "<TOOLCHAIN>\n"
    for d in detected:
        result += f"  Language: {d['lang']}\n"
        result += f"  Package Manager: {d['pkg']}\n"
        result += f"  Formatter: {d['fmt']}\n"
        result += f"  Test Runner: {d['test']}\n"
        if len(detected) > 1:
            result += "  ---\n"
    result += "</TOOLCHAIN>"
    print(f"  🔧 Toolchain: {', '.join(d['lang'] for d in detected)}", flush=True)
    return result


def get_symbol_skeleton(project_dir: str) -> str:
    """Matches class/function signatures AND imports to create a navigable project map."""
    import re
    skeleton = ["[PROJECT SYMBOL SKELETON]"]
    signature_re = re.compile(
        r"^\s*(?:class|def|function|interface|type|async\s+function)\s+([a-zA-Z0-9_]+)",
        re.MULTILINE
    )
    import_re = re.compile(
        r"^\s*(?:import\s+.+|from\s+\S+\s+import\s+.+|#include\s+.+|require\(.+\))",
        re.MULTILINE
    )

    total_chars = 0
    MAX_SKELETON_CHARS = 15000
    SKIP_DIRS = {"node_modules", ".git", "venv", ".venv", "__pycache__",
                 "dist", "build", "public", ".knowledge_base", ".cline_context", ".cline_logs"}

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            if file.endswith((".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp", ".h")):
                rel_path = os.path.relpath(os.path.join(root, file), project_dir)
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()

                    imports = import_re.findall(content)
                    symbols = signature_re.findall(content)
                    line_count = content.count('\n') + 1

                    if imports or symbols:
                        file_block = [f"\n{rel_path} ({line_count} lines)"]

                        if imports:
                            file_block.append("  imports:")
                            for imp in imports[:5]:
                                file_block.append(f"    {imp.strip()}")
                            if len(imports) > 5:
                                file_block.append(f"    ... +{len(imports)-5} more")

                        if symbols:
                            file_block.append("  defines:")
                            for sym in symbols:
                                file_block.append(f"    - {sym}")

                        block_str = "\n".join(file_block)
                        if total_chars + len(block_str) > MAX_SKELETON_CHARS:
                            skeleton.append("\n... [Skeleton truncated]")
                            return "\n".join(skeleton)
                        skeleton.append(block_str)
                        total_chars += len(block_str)
                except Exception:
                    continue
    return "\n".join(skeleton)


def run_distillation():
    """Execute the 4-pass distillation pipeline."""
    print("=" * 60, flush=True)
    print("🧠 Multi-Pass Context Distillation Engine", flush=True)
    print("=" * 60, flush=True)

    config = load_config()
    models = config.get("models", {})
    prompts = config.get("prompts", {})
    messages = load_conversation()

    def read_workspace_file(rel_path: str) -> str:
        """Helper to read a file from the workspace if it exists."""
        full_path = os.path.join("/workspace", rel_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    has_git = os.path.exists("/workspace/.git")
    has_code = any(f for f in os.listdir("/workspace") if f not in [".cline_context", ".cline_logs", ".knowledge_base", "conversation.json"])
    is_rebuild = os.path.exists(OUTPUT_PATH) or has_git or has_code

    if is_rebuild:
        status_text = "ALREADY PARTIALLY IMPLEMENTED" if not has_git else "EXISTING REPOSITORY DETECTED"
        print(f"\n🔄 {status_text} for {PROJECT_NAME}: Using structured context and latest instruction.", flush=True)
        import subprocess
        try:
            tree_output = subprocess.check_output(
                ["tree", "/workspace", "-I", "node_modules|.git|venv|.venv|.cline_context|.cline_logs|__pycache__|dist|build|public|.knowledge_base"], 
                text=True, stderr=subprocess.DEVNULL
            )
        except Exception:
            tree_output = "(Could not generate directory tree)"

        symbol_skeleton = get_symbol_skeleton("/workspace")
        toolchain_info = detect_project_toolchain("/workspace")

        latest_instruction = ""
        user_directives = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and "!build" in msg.get("content", "").lower():
                content = msg.get("content", "")
                latest_instruction = content
                # Extract directives: remove !build and flags
                import re
                directives = re.sub(r'!build|--repo\s+\S+|--kb\s+\S+', '', content, flags=re.IGNORECASE).strip()
                if directives:
                    user_directives = f"\n  <USER_DIRECTIVES>\n{directives}\n  </USER_DIRECTIVES>\n"
                break

        if not latest_instruction and messages:
            latest_instruction = messages[-1].get("content", "")

        readme_content = read_workspace_file("README.md")
        issues_content = read_workspace_file(".cline_context/.build_issues.md")

        kb_content = ""
        kb_dir = "/workspace/.knowledge_base"
        if os.path.exists(kb_dir):
            kb_content = select_relevant_kb(kb_dir, latest_instruction)

        conversation_text = (
            f"<SITUATIONAL_AWARENESS>\n"
            f"  <MODE>ITERATIVE_REBUILD</MODE>\n"
            f"  <STATUS>This project is ALREADY PARTIALLY IMPLEMENTED. Use the provided DIRECTORY_STRUCTURE and SYMBOL_SKELETON to understand the current state.</STATUS>\n"
            f"  <DIRECTIVES>\n"
            f"    1. [P0] PRESERVATION: Prioritize building on top of existing code. Maintain the current file organization and design idioms. Rework is strictly prohibited.\n"
            f"    2. [P0] CONTINUITY: Read the 'PROJECT_HISTORY' to pick up exactly where the last agent left off.\n"
            f"    3. [P0] NAVIGATION: Use the SYMBOL_SKELETON to map out dependencies before reading files.\n"
            f"    4. [P0] ANALYZE: Carefully examine the 'DIRECTORY_STRUCTURE', 'SYMBOL_SKELETON', and 'PROJECT_OVERVIEW' blocks below before planning any code changes.\n"
            f"    5. [P0] NON-REDUNDANT_PLANNING: DO NOT plan for or recreate files that already exist in the structure unless the 'NEW_REQUEST' explicitly requires a logic change in them.\n"
            f"    6. [P0] FILE_STATUS_AWARENESS: If the 'ARCHITECTURE' section (developed by the architect) mentions a file that is NOT present in the 'DIRECTORY_STRUCTURE', it is a NEW component. You MUST create it.\n"
            f"    7. [P0] CONTEXT_ALIGNMENT: Use the 'PROJECT_HISTORY' to understand the intent and reasoning behind the current request.\n"
            f"    8. [P1] SCOPE_FOCUS: Focus exclusively on fulfilling the 'NEW_REQUEST' and resolving the 'KNOWN_BUILD_ISSUES'.\n"
            f"  </DIRECTIVES>\n"
            f"{user_directives}"
            f"</SITUATIONAL_AWARENESS>\n\n"
            f"<PROJECT_DATA>\n"
            f"  <NAME>{PROJECT_NAME}</NAME>\n"
            f"  <PROJECT_HISTORY>\n"
            f"{conversation_to_text(messages[:-1])}\n"
            f"  </PROJECT_HISTORY>\n\n"
        )

        if kb_content:
            conversation_text += f"\n<BEST_PRACTICES_KNOWLEDGE_BASE>\n{kb_content}\n</BEST_PRACTICES_KNOWLEDGE_BASE>\n\n"

        if readme_content:
            conversation_text += f"  <PROJECT_OVERVIEW>\n```markdown\n{readme_content}\n```\n  </PROJECT_OVERVIEW>\n\n"

        if issues_content:
            conversation_text += f"  <KNOWN_BUILD_ISSUES>\n```markdown\n{issues_content}\n```\n  </KNOWN_BUILD_ISSUES>\n\n"

        conversation_text += (
            f"  <DIRECTORY_STRUCTURE>\n```\n{tree_output}\n```\n  </DIRECTORY_STRUCTURE>\n\n"
            f"  <SYMBOL_SKELETON>\n{symbol_skeleton}\n  </SYMBOL_SKELETON>\n\n"
        )
        if toolchain_info:
            conversation_text += f"  {toolchain_info}\n\n"
        conversation_text += (
            f"  <NEW_REQUEST>\n{latest_instruction}\n  </NEW_REQUEST>\n"
            f"</PROJECT_DATA>"
        )
    else:
        print(f"\n✨ Fresh build detected for {PROJECT_NAME}. Assembling historical context...", flush=True)

        # Assemble the historical conversation
        history = []
        final_command = ""

        # We assume the last message containing !build is the 'trigger'
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if "!build" in msg.get("content", "").lower():
                final_command = msg.get("content", "")
                history = messages[:i] # Everything before the trigger
                break

        if not final_command and messages:
            final_command = messages[-1].get("content", "")
            history = messages[:-1]

        conversation_text = (
            f"<SITUATIONAL_AWARENESS>\n"
            f"  <MODE>FRESH_BUILD</MODE>\n"
            f"  <STATUS>This is a NEW PROJECT START. Establish the foundational structure and implementation plan.</STATUS>\n"
            f"  <DIRECTIVES>\n"
            f"    1. [P0] FOUNDATION: Read the 'PROJECT_HISTORY' to understand the core vision, tech stack, and requirements.\n"
            f"    2. [P0] PRESERVATION: Maintain consistency with any existing patterns established in the project history.\n"
            f"    3. [P0] EXECUTION: Treat the 'FINAL_BUILD_COMMAND' as your immediate tactical mission.\n"
            f"    4. [P1] ALIGNMENT: Ensure your output fulfills both the historical vision and the final instruction based strictly on your assigned system role.\n"
            f"    5. [P1] CLARITY: Ensure the foundational structure is clean and well-documented.\n"
            f"  </DIRECTIVES>\n"
            f"</SITUATIONAL_AWARENESS>\n\n"
            f"<PROJECT_DATA>\n"
            f"  <NAME>{PROJECT_NAME}</NAME>\n"
            f"  <PROJECT_HISTORY>\n"
            f"{conversation_to_text(history)}\n"
            f"  </PROJECT_HISTORY>\n\n"
            f"  <FINAL_BUILD_COMMAND>\n{final_command}\n  </FINAL_BUILD_COMMAND>\n"
            f"</PROJECT_DATA>"
        )

    print(f"📄 Context size: {len(conversation_text)} chars", flush=True)

    passes = [
        ("architect",     "🏗️  Pass 1/4: System Architect"),
        ("engineer",      "⚙️  Pass 2/4: Engineer"),
        ("test_engineer", "🧪  Pass 3/4: Test Engineer"),
        ("safety",        "🛡️  Pass 4/4: Safety Inspector"),
    ]

    results = {}
    previous_model_config = None

    with httpx.Client() as client:
        for pass_key, pass_label in passes:
            print(f"\n{pass_label}", flush=True)
            print("-" * 40, flush=True)
            update_status(f"Distilling: {pass_label}")

            model_entry = models.get(pass_key, models.get("architect"))
            model_config = _resolve_model_config(model_entry)
            model_name = model_config.get("model", "")
            raw_prompt = prompts.get(pass_key, "Analyze the following conversation.")
            prompt = resolve_prompt(raw_prompt, CONFIG_PATH)

            # Check if we need to switch models (compare by model name)
            prev_name = previous_model_config.get("model", "") if previous_model_config else None
            if prev_name and prev_name != model_name:
                print(f"  ↳ Switching model: {prev_name} → {model_name} ({model_config.get('provider', 'ollama')})")
                unload_model(client, previous_model_config)

            # Pre-load non-Ollama models via orchestrator
            if model_config.get("provider", "ollama") != "ollama":
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        print(f"  ↳ Pre-loading model ({model_config.get('provider')}): {model_name} (Attempt {attempt+1}/{max_retries})")
                        # Pass ctx_size so orchestrator spawns llama-server with correct -c flag
                        load_payload = {**model_config, "ctx_size": CONTEXT_WINDOW}
                        client.post(
                            f"{ORCHESTRATOR_URL}/internal/model/load",
                            json=load_payload,
                            timeout=120.0
                        )
                        break # Success
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"  ⚠ Pre-load retryable error: {e}. Retrying in 5s...")
                            time.sleep(5)
                        else:
                            print(f"  ❌ Pre-load failed after {max_retries} attempts: {e}")
                            if "101" in str(e) or "Network is unreachable" in str(e):
                                print("    TIP: This usually means the host orchestrator is restarting the model. Check orchestrator.log on host.")

            prior_context = ""
            if results:
                prior_context = "\n\n".join(
                    f"#### {k.upper()} ANALYSIS\n{v}"
                    for k, v in results.items()
                )

            if pass_key in ["architect", "engineer"]:
                target_content = conversation_text
            else:
                # Passes 3 and 4 only read the previous plans. They do not get chunked!
                target_content = "Review the PREVIOUS ANALYSES provided above based strictly on your system role and required template format. Do not invent new features or write source code."

            result = call_llm(client, model_config, prompt, target_content, prior_context)
            results[pass_key] = result
            previous_model_config = model_config
            print(f"  ✓ Complete ({len(result)} chars)")

            intermediate_path = f"/workspace/.cline_context/distill_{pass_key}.md"
            try:
                # Ensure context directory exists inside workspace in case running raw
                os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
                with open(intermediate_path, "w", encoding="utf-8") as f:
                    f.write(f"# Distillation Intermediate: {pass_key.title()}\n\n{result}")
                print(f"  ↳ Saved intermediate result to {intermediate_path}")
            except Exception as e:
                print(f"  ⚠ Failed to save intermediate result: {e}")

        # We keep model warm for Phase 2 (the Cline Build cycle).

    print(f"\n📝 Writing {OUTPUT_PATH}", flush=True)
    update_status("Assembling .clinerules...")
    clinerules = assemble_clinerules(results, config, messages)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(clinerules)

    update_status("Distillation complete.")
    print(f"  ✓ Written ({len(clinerules)} chars)", flush=True)
    print("=" * 60, flush=True)
    print("✅ Distillation complete", flush=True)
    print("=" * 60, flush=True)


def assemble_clinerules(results: dict, config: dict, messages: list) -> str:
    """Combine the 4-pass results into a structured .clinerules document."""
    limits = config.get("limits", {})

    # Try to extract the target objective from the messages
    target_obj = "Complete the implementation roadmap as specified."
    for msg in reversed(messages):
        if "!build" in msg.get("content", "").lower():
            import re
            content = msg.get("content", "")
            # Filter out !build and flags
            target_obj = re.sub(r'!build|--repo\s+\S+|--kb\s+\S+', '', content, flags=re.IGNORECASE).strip()
            if not target_obj:
                target_obj = "Process project requirements and implement planned architecture."
            break

    doc = [
        "# Project Build Specification",
        "",
        "## 🎯 Current Target Objective",
        f"> {target_obj}",
        "",
        "> Auto-generated by Multi-Agent Distillation Pipeline",
        f"> Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## ⚠️ CORE DIRECTIVES (High Priority)",
        "The following requirements are non-negotiable and must be prioritized over all other implementation details:",
        "",
    ]

    if "test_engineer" in results:
        doc.append("### 🧪 Critical Test & Quality Gates")
        doc.append(results["test_engineer"])
        doc.append("")

    if "safety" in results:
        doc.append("### 🛡️ Safety & Security Mitigations")
        doc.append(results["safety"])
        doc.append("")

    doc.extend([
        "<operational_constraints>",
        f"- Max project size: {limits.get('max_project_size_mb', 4096)} MB",
        f"- Max build iterations: {limits.get('max_build_iterations', 5)}",
        "- PRESERVATION POLICY: Prioritize building on top of existing code. Maintain the current file structure and design patterns. Unsolicited rework, file-splitting, or structural optimization is strictly forbidden.",
        "- WEB RESEARCH TOOLS: To get latest documentation or libraries, use 'web-search \"<query>\"' in terminal. To inspect documentation pages, use 'fetch-page \"<url>\"'.",
        "- ANTI-LOOP RULE: Never attempt the same bug fix more than twice.",
        "- FOCUS REMINDER: Keep the main goal in mind. Do not get distracted by hypothetical features.",
        "- TASK COMPLETION: Relentlessly work through your checklist. Mark impossible tasks as blocked and move on.",
        "- If a test fails repeatedly, comment it out, add a TODO.",
        "- Finishing the checklist is more important than passing every test.",
        "- Verify each major component after implementation.",
        "- Run all safety checks before declaring the build complete.",
        "- CRITICAL CONTEXT RULE: NEVER search, read, or modify `node_modules/`, `.git/`, `__pycache__/` or `.venv/`.",
        "- PORT MANAGEMENT: If a port is in use, YOU MUST ONLY use `npx kill-port <port>` to free it. NEVER use pkill or kill commands.",
        "- DAEMON EXECUTION (CRITICAL): NEVER run `python3 -m http.server`, `npm start`, or ANY server command directly. It will hang the terminal and break the pipeline. You MUST use background processes: `python3 -m http.server 8000 &` or `nohup npm start &`.",
        "- REASONING: Before executing any terminal command or modifying files, you must write out a brief step-by-step logical analysis of your plan.",
        "- CONTEXT PRESERVATION: Your context window is limited. NEVER read more than 300 lines at once. Use searchFiles to locate specific code before reading.",
        "- EXTERNAL MEMORY: After analyzing any file, append a 3-line summary to '.cline_context/analysis_notes.md'. This is your long-term memory.",
        "- ANTI-AMNESIA: If you feel lost or unsure what you've done, read '.cline_context/.session_state.md' and '.cline_context/analysis_notes.md' BEFORE doing anything else.",
        "- SYMBOL SKELETON FIRST: Your .clinerules contains a Symbol Skeleton with imports and function names. Use this to navigate, not readFile.",
        "- DEBUG-FIRST: When you need to understand how code works, write a small probe script, run it, and read the output. This is faster and more accurate than reading 500 lines of source code.",
        "- MANDATORY TEST GATE: After editing ANY file, run the project's test suite. If tests fail after your edit, fix the regression BEFORE moving to the next task.",
        "</operational_constraints>",
        "",
    ])

    section_map = {
        "architect": ("Architecture & Directory Structure", "🏗️"),
        "engineer": ("Implementation Roadmap", "⚙️"),
    }

    for key, (title, icon) in section_map.items():
        if key in results:
            doc.extend([
                f"## {icon} {title}",
                "",
                results[key],
                "",
            ])

    return "\n".join(doc)


if __name__ == "__main__":
    run_distillation()
