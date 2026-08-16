import asyncio
import json
import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import mover

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Bob-Orchestrator")

# --- STRICT MODEL CONFIG ---
# Each model role is a dict: model name, provider type, and API base URL.
# Supported providers: "ollama", "lmstudio", "llamacpp"
EXPERT_CONFIG = {
    "model": "qwen3.8opt:latest",
    "provider": "ollama",
    "base_url": "http://localhost:11434",
}
ROUTER_CONFIG = {
    "model": "qwen2.5:1.5b",
    "provider": "ollama",
    "base_url": "http://localhost:11434",
}

# Shorthand names (derived from config, used in routing logic throughout)
EXPERT_MODEL = EXPERT_CONFIG["model"]
ROUTER_MODEL = ROUTER_CONFIG["model"]
DEFAULT_EXPERT_MODEL = "qwen3.8opt:latest" # Keep this as default, if you know what youre doing, you can change it with the PARAMS_GENERAL/PARAMS_CODING

# llama.cpp managed process settings (only used when provider is "llamacpp")
LLAMACPP_BINARY = "/home/mv/llama.cpp/build/bin/llama-server"  # Path to llama-server binary
LLAMACPP_DEFAULT_ARGS = ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]  # Extra CLI args (KV Quant enabled)

COMFYUI_URL = "http://localhost:8188"
EXPERT_CTX = 65536    # Context for the expert model (65k)
DISTILL_CTX = 65536   # Context for the distillation engine (65k)
CLINE_CTX = 65536     # Context for the Cline agent (65k)

# --- STATE MANAGEMENT ---
gpu_lock = asyncio.Lock()
http_client: httpx.AsyncClient = None
vram_locked = False
expert_warm_until = 0
expert_mode = "general"  # "general" or "coding"
last_comfy_history_count = 0  # Track ComfyUI history count for automated pings

# --- EXPERT PARAMETERS (Thinking Modes) ---
PARAMS_GENERAL = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
}

PARAMS_CODING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.15,
}

# =============================================================================
# PROVIDER MANAGEMENT
# =============================================================================
# Tracks managed llama-server subprocesses keyed by model name.
_managed_processes: dict[str, subprocess.Popen] = {}
# Locks to prevent multiple concurrent startups of the same llama-server.
_llamacpp_locks: dict[str, asyncio.Lock] = {}
# Events to signal when a llama-server is healthy and ready for requests.
_llamacpp_ready_events: dict[str, asyncio.Event] = {}
# Debounce timer for !build commands to prevent duplicate triggers from client retries.
_last_build_trigger_time = 0.0


def _resolve_config(model_name: str) -> dict:
    """Resolve a model name to its full provider configuration."""
    if model_name == EXPERT_MODEL:
        return EXPERT_CONFIG
    elif model_name == ROUTER_MODEL:
        return ROUTER_CONFIG
    # Fallback: assume Ollama on default port
    return {"model": model_name, "provider": "ollama", "base_url": "http://localhost:11434"}


def _is_ollama_provider(config: dict) -> bool:
    """Check if a model config uses the Ollama provider."""
    return config.get("provider", "ollama") == "ollama"


def _get_base_url(config: dict) -> str:
    """Get the API base URL from a model config."""
    return config.get("base_url", "http://localhost:11434")


def _get_chat_url(config: dict, prefer_native: bool = False) -> str:
    """
    Get the full chat endpoint URL for a provider.
    Ollama supports both /api/chat (native) and /v1/chat/completions.
    All other providers use /v1/chat/completions exclusively.
    """
    base = _get_base_url(config)
    if _is_ollama_provider(config) and prefer_native:
        return f"{base}/api/chat"
    return f"{base}/v1/chat/completions"


def _adapt_body(body: dict, config: dict) -> dict:
    """
    Adapt an Ollama-format request body for the target provider.
    Ollama uses an 'options' dict; OpenAI-compatible providers use top-level params.
    Returns the body unchanged for Ollama, or a translated copy for others.
    """
    if _is_ollama_provider(config):
        return body

    adapted = {
        "model": body.get("model", config.get("model", "")),
        "messages": body.get("messages", []),
        "stream": body.get("stream", True),
    }

    # Map Ollama `options` to top-level OpenAI parameters
    options = body.get("options", {})
    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        if key in options:
            adapted[key] = options[key]

    if body.get("format") == "json":
        adapted["response_format"] = {"type": "json_object"}

    if "tools" in body:
        adapted["tools"] = body["tools"]

    return adapted


def _prune_messages(messages: list, max_chars: int = 90000) -> list:
    """
    In-place pruner that ensures the conversation remains below a safe threshold.
    Preserves the system message (first) and the most recent turns (last 4).
    """
    if not messages or len(messages) <= 5:
        return messages

    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    if total_chars <= max_chars:
        return messages

    logger.info(f"[Safety Monitor] Pruning context: {total_chars} chars > {max_chars} limit.")

    # Keep system message and last 4 turns
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    footer = messages[-4:]

    # Prune from the middle (starting after system message)
    middle = messages[1:-4] if system_msg else messages[:-4]

    # Simple strategy: Keep removing oldest middle messages until under limit
    while total_chars > max_chars and middle:
        removed = middle.pop(0)
        total_chars -= len(str(removed.get("content", "")))

    new_messages = []
    if system_msg:
        new_messages.append(system_msg)
    new_messages.extend(middle)
    new_messages.extend(footer)

    logger.info(f"[Safety Monitor] New context size: {sum(len(str(m.get('content', ''))) for m in new_messages)} chars.")
    return new_messages


def _docker_to_host_url(url: str) -> str:
    """Convert Docker container URLs to host-accessible URLs."""
    return url.replace("host.docker.internal", "localhost")


async def _provider_load(config: dict):
    """Ensure a model is loaded and ready for its provider."""
    provider = config.get("provider", "ollama")
    model = config.get("model", "")

    if provider == "ollama":
        return  # Ollama auto-loads on first request

    elif provider == "lmstudio":
        try:
            logger.info(f"LM Studio: Loading {model}...")
            proc = await asyncio.create_subprocess_exec(
                "lms", "load", model, "-y",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                logger.info(f"LM Studio: Loaded {model}")
            else:
                logger.warning(f"LM Studio: Load may have failed: {stderr.decode().strip()}")
        except FileNotFoundError:
            logger.error("LM Studio: 'lms' CLI not found. Install with: ~/.lmstudio/bin/lms bootstrap")
        except Exception as e:  # noqa: BLE001
            logger.error(f"LM Studio: Load error: {e}")

    elif provider == "llamacpp":
        await _start_llamacpp_server(config)


async def _start_llamacpp_server(config: dict):
    """Start a managed llama-server process and wait for readiness."""
    model = config.get("model", "")
    if not model:
        return

    # Ensure state exists for this model
    if model not in _llamacpp_locks:
        _llamacpp_locks[model] = asyncio.Lock()
    if model not in _llamacpp_ready_events:
        _llamacpp_ready_events[model] = asyncio.Event()

    # Get the ready event early
    ready_event = _llamacpp_ready_events[model]

    async with _llamacpp_locks[model]:
        # Check if already running or starting
        if model in _managed_processes:
            proc = _managed_processes[model]
            if proc.poll() is None:
                # Still alive. Is it ready?
                if ready_event.is_set():
                    logger.info(f"llama.cpp: {model} is already running and ready.")
                    return
                else:
                    logger.info(f"llama.cpp: {model} is already starting. Waiting for readiness...")
            else:
                logger.info(f"llama.cpp: Previous process for {model} died. Restarting...")
                del _managed_processes[model]
                ready_event.clear()

        # Parse config for startup
        base_url = config.get("base_url", "http://localhost:8081")
        extra_args = list(config.get("args", LLAMACPP_DEFAULT_ARGS))
        binary = config.get("binary_path", LLAMACPP_BINARY)

        # Parse host/port from base_url
        parsed = urlparse(base_url)
        port = str(parsed.port or 8081)
        # Force bind to 0.0.0.0 so Docker containers (like cline-builder) can connect
        # via host.docker.internal, otherwise 127.0.0.1 blocks them.
        host = "0.0.0.0"

        # Context window: use config override, or fall back to EXPERT_CTX
        ctx_size = str(config.get("ctx_size", EXPERT_CTX))

        # Context rotation: llama-server natively supports sliding window truncation
        # Set to 0.7 to give massive breathing room and reduce rotation frequency.
        # Disable similarity matching to force linear slot persistence.
        truncate_args = [
            "--chat-truncate",
            "--chat-truncate-max-keep", "0.7",
            "--slot-prompt-similarity", "0.95",
            "--batch-size", "1024",
            "--ubatch-size", "1024"
        ]

        # Build command: detect HuggingFace repo vs local path
        if "/" in model and not os.path.exists(model):
            cmd = [binary, "-hf", model, "--host", host, "--port", port, "-c", ctx_size, "-np", "1"] + truncate_args + extra_args
        else:
            cmd = [binary, "-m", model, "--host", host, "--port", port, "-c", ctx_size, "-np", "1"] + truncate_args + extra_args

        if model not in _managed_processes:
            logger.info(f"llama.cpp: Starting: {' '.join(cmd)}")
            ready_event.clear()

            # Force Flash Attention ON to save VRAM and improve speed at 128k context
            env = os.environ.copy()
            env["LLAMA_ARG_FLASH_ATTN"] = "on"

            log_file = open("llama-server.log", "a")  # noqa: ASYNC230, SIM115
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)  # noqa: ASYNC220
            _managed_processes[model] = proc

            # Start background health-check task
            asyncio.create_task(_poll_llamacpp_health(model, base_url, ready_event))

    # Wait for the background task to signal readiness
    # This happens OUTSIDE the lock, so other requests can enter and 'wait' for the same event
    try:
        await asyncio.wait_for(ready_event.wait(), timeout=120.0)
        logger.info(f"llama.cpp: {model} is confirmed ready.")
    except asyncio.TimeoutError:
        logger.error(f"llama.cpp: Timeout waiting for {model} readiness.")
        raise TimeoutError(f"Model startup timed out: {model}") from None


async def _poll_llamacpp_health(model: str, base_url: str, ready_event: asyncio.Event):
    """Background task to poll a model's health and signal readiness."""
    deadline = time.time() + 120
    while time.time() < deadline:
        # Verify the process is still running
        if model in _managed_processes:
            proc = _managed_processes[model]
            if proc.poll() is not None:
                logger.error(f"llama.cpp: Process for {model} died during startup (code {proc.returncode})")
                return

        try:
            resp = await http_client.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                logger.info(f"llama.cpp: Background health-check passed for {model}")
                ready_event.set()
                return
        except (httpx.HTTPError, OSError):
            pass
        await asyncio.sleep(2.0)

    logger.error(f"llama.cpp: Background health-check TIMED OUT for {model}")


async def _stop_llamacpp_server(config: dict):
    """Stop a managed llama-server process and free its resources."""
    model = config.get("model", "")
    if model in _managed_processes:
        proc = _managed_processes[model]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            logger.info(f"llama.cpp: Stopped {model}")
        del _managed_processes[model]


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifecycle, initializing and closing the HTTP client."""
    global http_client, last_comfy_history_count
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    logger.info("HTTP client initialized.")

    # Synchronize initial ComfyUI history state
    last_comfy_history_count = await get_comfy_history_count()

    # Start periodic background cleanup
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield
    cleanup_task.cancel()
    await http_client.aclose()
    logger.info("HTTP client closed.")


app = FastAPI(title="Bob: AI Workspace Orchestrator", lifespan=lifespan)

# =============================================================================
# VRAM MANAGEMENT
# =============================================================================

async def get_loaded_models() -> list[str]:
    """Queries all configured providers to identify models currently in VRAM."""
    loaded = []
    checked_ollama_urls = set()

    for cfg in [EXPERT_CONFIG, ROUTER_CONFIG]:
        provider = cfg.get("provider", "ollama")
        base_url = _get_base_url(cfg)

        if provider == "ollama" and base_url not in checked_ollama_urls:
            checked_ollama_urls.add(base_url)
            try:
                resp = await http_client.get(f"{base_url}/api/ps", timeout=2.0)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    loaded.extend([m.get("name", "") for m in models])
            except (httpx.HTTPError, OSError, ValueError):
                pass

        elif provider == "lmstudio":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "lms", "ps",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                if cfg["model"] in stdout.decode():
                    loaded.append(cfg["model"])
            except (asyncio.SubprocessError, OSError):
                pass

    # Check managed llama.cpp processes
    for model_name, proc in list(_managed_processes.items()):
        if proc.poll() is None:
            loaded.append(model_name)
        else:
            del _managed_processes[model_name]

    return loaded

async def force_unload(model_name: str):
    """
    Unloads a model using the appropriate provider method.
    Resolves the provider from the model name automatically.
    """
    config = _resolve_config(model_name)
    provider = config.get("provider", "ollama")
    base_url = _get_base_url(config)

    try:
        if provider == "ollama":
            logger.info(f"Unloading model (Ollama): {model_name}")
            await http_client.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=5.0
            )
        elif provider == "lmstudio":
            logger.info(f"Unloading model (LM Studio): {model_name}")
            proc = await asyncio.create_subprocess_exec(
                "lms", "unload", model_name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
        elif provider == "llamacpp":
            logger.info(f"Unloading model (llama.cpp): {model_name}")
            await _stop_llamacpp_server(config)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to unload {model_name}: {e}")

async def verified_unload(model_name: str, max_wait: float = 3.0):
    """
    Attempts to unload a model and verifies its removal through short polling.
    For non-Ollama providers, unloads directly without polling.
    """
    config = _resolve_config(model_name)
    provider = config.get("provider", "ollama")

    if provider == "ollama":
        base_url = _get_base_url(config)
        # Check if currently loaded
        try:
            resp = await http_client.get(f"{base_url}/api/ps", timeout=2.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                names = [m.get("name", "") for m in models]
                if not any(model_name in m for m in names):
                    return True
        except (httpx.HTTPError, OSError, ValueError):
            pass

        await force_unload(model_name)

        # Poll for removal
        deadline = time.time() + max_wait
        while time.time() < deadline:
            await asyncio.sleep(0.4)
            try:
                resp = await http_client.get(f"{base_url}/api/ps", timeout=2.0)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    names = [m.get("name", "") for m in models]
                    if not any(model_name in m for m in names):
                        return True
            except (httpx.HTTPError, OSError, ValueError):
                pass

        logger.warning(f"{model_name} persistent in VRAM after {max_wait}s; continuing.")
        return False
    else:
        # Non-Ollama: unload and trust it worked
        await force_unload(model_name)
        await asyncio.sleep(0.5)
        return True


async def sweep_vram_for_expert():
    """
    Ensures the Router model is removed before loading the Expert model.
    This prevents VRAM fragmentation and avoids offloading to slower system RAM.
    """
    loaded = await get_loaded_models()
    if any(ROUTER_MODEL in m for m in loaded):
        logger.info("Sweeping VRAM for Expert model load.")
        await verified_unload(ROUTER_MODEL)

async def free_comfyui():
    """Immediately signals ComfyUI to clear models and release system/video memory."""
    try:
        await http_client.post(
            f"{COMFYUI_URL}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=3.0
        )
    except (httpx.HTTPError, OSError):
        pass


async def periodic_cleanup():
    """
    Background loop that periodically sweeps memory if the system is idle.
    Runs every 5 minutes. Also auto-unloads llama-server when the expert
    warm timer expires (mirrors Ollama's native keep_alive behavior).
    """
    while True:
        try:
            await asyncio.sleep(300)  # 5 minute interval
            if not await is_comfy_active() and not gpu_lock.locked():
                logger.info("Periodic idle cleanup triggered.")
                await free_comfyui()

                # Auto-unload llama-server when expert warm period expires
                if not vram_locked and time.time() > expert_warm_until:
                    expert_config = _resolve_config(EXPERT_MODEL)
                    if expert_config.get("provider") == "llamacpp" and EXPERT_MODEL in _managed_processes:
                        proc = _managed_processes[EXPERT_MODEL]
                        if proc.poll() is None:
                            logger.info(f"Expert warm period expired. Auto-stopping llama-server for {EXPERT_MODEL}.")
                            await _stop_llamacpp_server(expert_config)
        except asyncio.CancelledError:
            break
        except Exception as e:  # noqa: BLE001
            logger.error(f"Cleanup Task Error: {e}")
            await asyncio.sleep(60)  # Wait before retry on error


async def is_comfy_active() -> bool:
    """Checks the ComfyUI queue for active or pending generation jobs."""
    try:
        resp = await http_client.get(f"{COMFYUI_URL}/queue", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            return len(data.get("queue_running", [])) > 0 or len(data.get("queue_pending", [])) > 0
    except (httpx.HTTPError, OSError, ValueError):
        pass
    return False

async def get_comfy_history_count() -> int:
    """Retrieves the total number of completed prompts from the ComfyUI history."""
    try:
        resp = await http_client.get(f"{COMFYUI_URL}/history", timeout=2.0)
        if resp.status_code == 200:
            return len(resp.json())
    except (httpx.HTTPError, OSError, ValueError):
        pass
    return 0

# =============================================================================
# TRIAGE
# =============================================================================

async def analyze_request(messages: list) -> dict:
    """
    Performs request triage using the Router model to determine task complexity.
    Routes complex logic or coding tasks to the Expert model.
    """
    recent_msgs = [m for m in messages if m.get("role") == "user"][-2:]
    context_text = "\n".join([m.get("content", "") for m in recent_msgs])

    system_msg = (
        "You are the Triage AI for a workspace. Read the user's prompt and evaluate it. "
        "The 'Expert' AI is incredibly busy and expensive. You must only call the Expert if "
        "the complexity is above 6 (e.g., coding, deep logic, advanced math). "
        "Complexity 1-4: Small talk, greetings, simple facts. "
        "Complexity 5-10: Deep logic, advanced math, structural coding. "
        "Guess if the user will need follow-up questions to solve this task (true/false). "
        "Also determine if this is a 'coding' task (True if involves writing code, debugging, or technical WebDev logic). "
        "and if it requires a tool (True if it needs web search). "
        "Respond ONLY in pure JSON format: {\"complexity\": <int>, \"expect_followups\": <bool>, \"requires_tool\": <bool>, \"is_coding\": <bool>}"
    )

    router_config = _resolve_config(ROUTER_MODEL)

    if _is_ollama_provider(router_config):
        payload = {
            "model": ROUTER_MODEL,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": context_text[:1000]}
            ],
            "stream": False,
            "keep_alive": 0,
            "options": {"temperature": 0.0, "num_ctx": 2048}
        }
        url = f"{_get_base_url(router_config)}/api/chat"
    else:
        payload = {
            "model": ROUTER_MODEL,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": context_text[:1000]}
            ],
            "stream": False,
            "temperature": 0.0,
        }
        url = f"{_get_base_url(router_config)}/v1/chat/completions"

    try:
        await _provider_load(router_config)
        resp = await http_client.post(url, json=payload, timeout=10.0)
        if resp.status_code == 200:
            resp_data = resp.json()
            if _is_ollama_provider(router_config):
                content = resp_data.get("message", {}).get("content", "{}")
            else:
                content = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            data = json.loads(content)
            return {
                "complexity": int(data.get("complexity", 1)),
                "followups": bool(data.get("expect_followups", False)),
                "requires_tool": bool(data.get("requires_tool", False)),
                "is_coding": bool(data.get("is_coding", False))
            }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Router network/parsing error: {e}")

    return {"complexity": 1, "followups": False, "requires_tool": False, "is_coding": False}

# =============================================================================
# STREAMING
# =============================================================================

async def stream_proxy(url: str, body: dict, lock: asyncio.Lock,
                       is_native: bool = False, backend_is_ollama: bool = True,
                       request_headers: dict | None = None):
    """
    Proxies a streaming response from the backend AI model while managing VRAM locks.
    Handles format translation between Ollama and OpenAI-compatible backends/clients.
    
    Args:
        is_native: True if the CLIENT expects Ollama-native format.
        backend_is_ollama: True if the BACKEND sends Ollama-native format.
        request_headers: Headers from the initial request (for X-No-Scrub).
    """
    if request_headers is None:
        request_headers = {}
    lock_released = False
    turn_throughput = 0  # Character counter to detect infinite reasoning loops

    def _release():
        nonlocal lock_released
        if not lock_released:
            lock_released = True
            try:
                lock.release()
            except RuntimeError:
                pass

    # Backend sends native Ollama line-JSON only when both backend is Ollama AND
    # the URL targets /api/chat (indicated by the client requesting native format).
    backend_sends_native = backend_is_ollama and is_native

    try:
        async with http_client.stream("POST", url, json=body, timeout=600.0) as resp:
            if resp.status_code != 200:
                error_body = ""
                async for chunk in resp.aiter_bytes():
                    error_body += chunk.decode("utf-8", errors="ignore")
                error_msg = error_body[:200].strip() or "Backend unavailable"
                logger.error(f"Backend error ({resp.status_code}): {error_body[:500]}")
                if is_native:
                    yield f'{json.dumps({"model": "Bob", "message": {"role": "assistant", "content": f"⚠️ {error_msg}"}, "done": True})}\n'.encode()
                else:
                    err_chunk = {"id": "chatcmpl-Bob", "object": "chat.completion.chunk", "model": "Bob",
                                  "choices": [{"index": 0, "delta": {"content": f"⚠️ {error_msg}"}, "finish_reason": "stop"}]}
                    yield f"data: {json.dumps(err_chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                return

            line_iter = resp.aiter_lines().__aiter__()

            # Check if scrubbing is disabled via header
            scrub_disabled = request_headers.get("X-No-Scrub", "false").lower() == "true"

            # Non-destructive heartbeat logic
            next_line_task = asyncio.create_task(line_iter.__anext__())

            # Syntax Scrubber state
            content_buffer = ""
            lookahead_buffer = "" # For split-chunk tag reconstruction
            in_tool_tag = False
            in_progress_block = False
            active_tag_name = ""
            seen_openers = set() # Track starting tags to prevent dangling closures

            while True:
                try:
                    # Wait for next line or heartbeat timeout (60s)
                    done, _ = await asyncio.wait(
                        [next_line_task],
                        timeout=60.0,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    if next_line_task not in done:
                        # Timeout! Send keeping-alive pulse
                        yield b": heartbeat\n\n"
                        continue

                    # Task is done, get the line
                    try:
                        line = next_line_task.result()
                        # Prepare the next line task immediately
                        next_line_task = asyncio.create_task(line_iter.__anext__())
                    except StopAsyncIteration:
                        break

                    if not line:
                        continue
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Iterator error: {e}")
                    break

                if backend_sends_native:
                    # Path 1: Ollama backend -> Native client (direct proxy)
                    try:
                        data = json.loads(line)
                        data["model"] = "Bob"
                        yield f"{json.dumps(data)}\n".encode()
                    except (json.JSONDecodeError, TypeError, KeyError):
                        yield f"{line}\n".encode()
                else:
                    # Backend sends OpenAI SSE format (data: {json})
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        if is_native:
                            yield f'{json.dumps({"model": "Bob", "message": {"role": "assistant", "content": ""}, "done": True})}\n'.encode()
                        else:
                            yield b"data: [DONE]\n\n"
                        continue
                    try:
                        data = json.loads(line[6:])

                        # VALIDATION: Discard chunks that don't have choices
                        if not is_native and "choices" not in data and not line.endswith("[DONE]"):
                            continue

                        data["model"] = "Bob"
                        if "id" in data:
                            data["id"] = "chatcmpl-Bob"

                        # Syntax Scrubber: Handle IQ4 'nested tag' hallucinations
                        choices = data.get("choices", [])
                        content = ""
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")

                            # Safety Monitor: 100k capacity for complex systems engineering.
                            # We track this REGARDLESS of scrubbing status to protect Distillation passes.
                            if content and not in_tool_tag:
                                turn_throughput += len(content)
                                if turn_throughput > 100000:
                                    logger.error(f"[Safety Monitor] Analytical capacity exceeded ({turn_throughput} chars). Triggering stability protocol.")
                                    # Inject closing tag if we are inside a tool call to prevent UI crash
                                    closure = ""
                                    if in_tool_tag and active_tag_name:
                                        logger.warning(f"[Safety Monitor] Force-closing tag </{active_tag_name}> due to capacity limit.")
                                        closure = f"</{active_tag_name}>\n\n[STABILITY MONITOR: Analytical capacity reached. Turn was terminated while '{active_tag_name}' was active. DO NOT re-read this file/data from the beginning. Use 'grep' or continue from line 1000+ if needed.]\n\n"

                                    delta["content"] = content + closure + "[STABILITY PROTOCOL: Analytical reasoning capacity exceeded (100k). Forcing turn completion.]"

                                    # Fragment Recovery: If server ends turn prematurely while tag is open
                                    finish_reason = choices[0].get("finish_reason")
                                    if (finish_reason == "length" or (is_native and data.get("done"))) and in_tool_tag and active_tag_name:
                                        logger.warning(f"[Safety Monitor] Server-side cutoff detected. Force-closing </{active_tag_name}>.")
                                        delta["content"] = (delta.get("content", "") or "") + f"</{active_tag_name}>\n\n[STABILITY PROTOCOL: Analytical capacity reached context limit. Session state preserved.]"
                                        # Ensure finish_reason is still set correctly for UI
                                        if not is_native:
                                            data["choices"][0]["finish_reason"] = "length"
                                        else:
                                            data["done"] = True

                                    if is_native:
                                        native_chunk = {"model": "Bob", "message": delta, "done": data.get("done", False)}
                                        yield f"{json.dumps(native_chunk)}\n".encode()
                                    else:
                                        yield f"data: {json.dumps(data)}\n\n".encode()
                                    return

                            if content and not scrub_disabled:
                                # Look-Ahead Reconstruction: Handle tags split across chunks
                                raw_content = lookahead_buffer + content
                                lookahead_buffer = ""

                                # If chunk ends with a partial tag, hold it for the next packet
                                if "<" in raw_content and ">" not in raw_content[raw_content.rfind("<"):]:
                                    split_idx = raw_content.rfind("<")
                                    # Limit lookahead to avoid stalling on just a solitary '<'
                                    if len(raw_content) - split_idx < 100:
                                        lookahead_buffer = raw_content[split_idx:]
                                        content = raw_content[:split_idx]
                                        # IMPORTANT: Clear outgoing content so we don't send the partial tag yet
                                        delta["content"] = content
                                    else:
                                        content = raw_content
                                else:
                                    content = raw_content

                                # UNIVERSAL FIELD SCRUBBER: Sanitize all delta fields (content, reasoning_content, thought, etc.)
                                scrub_targets = ["thinking", "thought", "thought_process", "read_file", "write_to_file", "execute_command"]
                                for key, value in list(delta.items()):
                                    if isinstance(value, str) and value:
                                        # Track openers in this Turn/Request
                                        for opener in re.findall(r'<\s*([a-zA-Z0-9_]+)\s*>', value):
                                            seen_openers.add(opener.lower())

                                        # Identify and strip dangling closures
                                        closures = re.findall(r'</\s*([a-zA-Z0-9_]+)\s*>', value, re.IGNORECASE)
                                        for closure in closures:
                                            if closure.lower() not in seen_openers and closure.lower() in scrub_targets:
                                                logger.warning(f"[Scrubber] Stripped dangling tag from delta['{key}']: </{closure}>")
                                                value = re.sub(f'</\\s*{re.escape(closure)}\\s*>', "", value, flags=re.IGNORECASE)

                                        delta[key] = value

                                # Standard content extraction for tool-state tracking
                                content = delta.get("content", "")

                                if in_tool_tag:
                                    if "<task_progress>" in content:
                                        in_progress_block = True
                                        content_buffer += content
                                        delta["content"] = " "
                                    elif "</task_progress>" in content:
                                        in_progress_block = False
                                        content_buffer += content
                                        delta["content"] = " "
                                    elif active_tag_name and f"</{active_tag_name}>" in content:
                                        if content_buffer:
                                            clean_progress = content_buffer.replace("<task_progress>", "### Progress:\n").replace("</task_progress>", "")
                                            delta["content"] = content + "\n\n" + clean_progress
                                            content_buffer = ""
                                        else:
                                            delta["content"] = content
                                        in_tool_tag = False
                                        active_tag_name = ""
                                    elif in_progress_block:
                                        content_buffer += content
                                        delta["content"] = " "
                                    else:
                                        delta["content"] = content
                                else:
                                    tag_match = re.search(r'<([a-z0-9_]+)>', content)
                                    if tag_match:
                                        tag_name = tag_match.group(1)
                                        if tag_name in ["read_file", "write_to_file", "search_files", "execute_command",
                                                      "list_files", "list_dir", "grep_search", "read_browser_page",
                                                      "replace_in_file", "insert_content", "ask_followup_question", "attempt_completion"]:
                                            in_tool_tag = True
                                            active_tag_name = tag_name
                                            logger.info(f"[Scrubber] Detected tool call across window: <{tag_name}>.")
                                    delta["content"] = content
                            elif content and scrub_disabled:
                                # Distillation path: Scrubber is off, but watchdog already tracked throughput above
                                delta["content"] = content

                        if is_native:
                            # Path 2: OpenAI backend -> Native client (translate)
                            token = delta.get("content", "")
                            done = choices[0].get("finish_reason") is not None
                            ollama_chunk = {
                                "model": "Bob", "message": {"role": "assistant", "content": token}, "done": done
                            }
                            yield f"{json.dumps(ollama_chunk)}\n".encode()
                        else:
                            # Path 3: OpenAI backend -> OpenAI client (direct proxy)
                            yield f"data: {json.dumps(data)}\n\n".encode()
                    except (json.JSONDecodeError, TypeError, KeyError):
                        rewritten = re.sub(r'("model"\s*:\s*")[^"]+(")', r'\1Bob\2', line)
                        yield f"{rewritten}\n\n".encode()

            # FINAL FLUSH: Handle dangling tags AND held lookahead fragments
            final_fix = ""
            if lookahead_buffer:
                final_fix += lookahead_buffer

            if in_tool_tag and active_tag_name:
                if content_buffer:
                    final_fix += "\n\n" + content_buffer.replace("<task_progress>", "### Progress:\n").replace("</task_progress>", "")
                final_fix += f"</{active_tag_name}>"
                logger.info(f"[Scrubber] Performing final auto-close for <{active_tag_name}>.")

            if final_fix:
                if is_native:
                    yield f'{json.dumps({"model": "Bob", "message": {"role": "assistant", "content": final_fix}, "done": True})}\n'.encode()
                else:
                    final_chunk = {"id": "chatcmpl-Bob", "object": "chat.completion.chunk", "model": "Bob",
                                  "choices": [{"index": 0, "delta": {"content": final_fix}, "finish_reason": "stop"}]}
                    yield f"data: {json.dumps(final_chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"

    except (GeneratorExit, asyncio.CancelledError):
        pass
    except Exception as e:  # noqa: BLE001
        logger.error(f"Stream proxy error: {e}")
        if is_native:
            yield f'{json.dumps({"model": "Bob", "message": {"role": "assistant", "content": f"⚠️ Error: {e!s}"}, "done": True})}\n'.encode()
        else:
            err_chunk = {"id": "chatcmpl-Bob", "object": "chat.completion.chunk", "model": "Bob",
                          "choices": [{"index": 0, "delta": {"content": f"\n\n⚠️ **Error:** {e!s}"}, "finish_reason": "error"}]}
            yield f"data: {json.dumps(err_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
    finally:
        _release()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/tags")
@app.get("/v1/models")
async def list_models(request: Request):
    """Returns a unified model list to satisfy both Ollama and OpenAI API clients."""
    is_tags = "tags" in str(request.url)
    bob_model = {
        "name": "Bob", "model": "Bob", "modified_at": "2026-03-16T00:00:00Z", "size": 0,
        "digest": "bob-identity",
        "details": {"family": "llama", "parameter_size": "Expert", "quantization_level": "Q8_0"}
    } if is_tags else {
        "id": "Bob", "object": "model", "created": int(time.time()), "owned_by": "System"
    }

    if is_tags:
        return JSONResponse(content={"models": [bob_model]})
    return JSONResponse(content={"object": "list", "data": [bob_model]})

@app.get("/health")
async def health_check():
    """Debug state."""
    loaded = await get_loaded_models()
    return JSONResponse(content={
        "loaded_models": loaded,
        "expert_warm": time.time() < expert_warm_until,
        "expert_mode": expert_mode,
        "vram_locked": vram_locked,
        "gpu_lock_held": gpu_lock.locked()
    })

@app.post("/v1/shutdown_expert")
async def shutdown_expert():
    """Explicitly shutdown any managed expert processes and release VRAM."""
    global expert_warm_until, vram_locked
    expert_warm_until = 0
    vram_locked = False

    killed_count = 0
    for model_name, proc in list(_managed_processes.items()):
        if proc.poll() is None:
            logger.info(f"Shutting down managed process for {model_name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except (subprocess.SubprocessError, OSError):
                proc.kill()
            killed_count += 1

        # Cleanup state
        _managed_processes.pop(model_name, None)
        if model_name in _llamacpp_ready_events:
            _llamacpp_ready_events[model_name].clear()

    try:
        # Use killall as a universal sweep for any binary name match
        subprocess.run(["killall", "-9", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)  # noqa: ASYNC221
        logger.info("VRAM Cleanup: Executed killall -9 llama-server sweep.")
    except (subprocess.SubprocessError, OSError):
        pass

    logger.info(f"VRAM Cleanup: Terminated {killed_count} tracked expert processes.")
    return JSONResponse(content={"status": "ok", "processes_killed": killed_count})

@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def proxy_ollama(request: Request):
    """
    Main entry point for AI chat requests.
    Handles orchestration, triage, interception, and VRAM management.
    """
    global vram_locked, expert_warm_until, expert_mode, last_comfy_history_count

    body = await request.json()
    is_streaming = body.get("stream", True)
    messages = body.get("messages", [])
    path = str(request.url.path)
    is_native = "/api/chat" in path

    is_background_task = False
    if messages:
        last_msg = messages[-1]
        last_content = str(last_msg.get("content", "")).lower()
        role = last_msg.get("role", "")

        # 1. Generation State Tracking
        is_busy = await is_comfy_active()
        current_history_count = await get_comfy_history_count()
        has_finished_gen = current_history_count > last_comfy_history_count
        has_gen_signal = is_busy or has_finished_gen

        # 2. Categorization
        # Detect UI-injected context tags
        system_content = "".join([m.get("content", "") for m in messages if m.get("role") == "system"]).lower()
        has_injected_context = "the requested image has been created" in system_content or "<context>" in system_content

        # Background tasks (Follow-ups, titles, etc.)
        is_suggestion_ping = any(kw in last_content for kw in ["suggest 3-5", "generate a title", "generate a short title", "summarize", "short label", "tags"])

        # Suppress suggestions immediately after a build launch to conserve VRAM for the 35B Expert
        if is_suggestion_ping and any("build pipeline" in str(m.get("content", "")).lower() for m in messages[-3:]):
            logger.info("Suppressing suggestion ping due to recent build pipeline trigger.")
            is_suggestion_ping = False

        # Prompt Expansion (Inhibited by policy)
        is_expansion_ping = "generate a detailed prompt" in last_content or "### task:\ngenerate a detailed prompt" in last_content

        # Image Descriptions
        description_keywords = ["describe", "analyze", "summarize", "tell me about", "what is in this", "what do you see", "explain the image"]
        is_description_ping = has_injected_context or (
            any(kw in last_content for kw in description_keywords) and ("### task:" in last_content or len(last_content) < 400)
        )

        # Tool results
        is_tool_result = (role == "tool")
        is_image_tool = is_tool_result and any(kw in last_content for kw in ["![", "comfy", "image_url", "image/"])
        is_search_tool = is_tool_result and not is_image_tool

        # Search Intent (In user prompt)
        search_triggers = [
            "google", "find on the web", "look up", "latest news", "recent news", "current weather",
            "what is the price of", "what is the status of", "current events", "today",
            "recently", "latest version of"
        ]
        is_search_query = (role == "user") and any(kw in last_content for kw in search_triggers)

        # Image Intent (In user prompt)
        image_triggers = ["generate an image", "create an image", "create a picture", "draw a", "make an image", "flux", "comfyui"]
        is_image_query = (role == "user") and any(kw in last_content for kw in image_triggers)

        # History-based Search Detection (Check only last 2 messages)
        has_search_history = any(
            (m.get("role") == "tool" or "retrieved" in str(m.get("content", "")).lower() or "sources" in str(m.get("content", "")).lower())
            and not any(kw in str(m.get("content", "")).lower() for kw in ["![", "comfy", "image_url", "image/"])
            for m in messages[-2:]
        )

        # 3. Interception Rules
        # Rule A: Silence image tool outputs
        if is_image_tool:
            last_comfy_history_count = current_history_count
            asyncio.create_task(free_comfyui())
            return _silent_response(is_native)

        # Rule B: Silence expansions and fresh-image descriptions
        should_kill_comment = (is_description_ping and has_gen_signal)
        if should_kill_comment or is_expansion_ping:
            last_comfy_history_count = current_history_count
            asyncio.create_task(free_comfyui())
            return _silent_response(is_native)

        # Rule C: Route background pings to small model
        is_background_task = (is_suggestion_ping or is_description_ping or is_expansion_ping) and not (is_search_tool or is_search_query or has_search_history or is_image_query)

        # Rule D: Signal Reset (Manual user turns clear the state)
        if (role == "user" and
            not (is_suggestion_ping or is_description_ping or is_expansion_ping or "### task:" in last_content) and
            current_history_count != last_comfy_history_count):
            last_comfy_history_count = current_history_count
            asyncio.create_task(free_comfyui())

        # Rule E: Suppress background tasks following maintenance commands
        maintenance_commands = ["!status", "!stop", "!move", "!build", "!lock", "!unlock"]
        is_maintenance_followup = False
        if is_background_task and len(messages) >= 3:
            prev_user_msg = messages[-3].get("content", "").lower() if messages[-3].get("role") == "user" else ""
            if any(cmd in prev_user_msg for cmd in maintenance_commands):
                is_maintenance_followup = True

        if is_maintenance_followup:
            logger.info("Maintenance follow-up detected: Silencing background task.")
            return _silent_response(is_native)

        # Rule F: Suppress background tasks during active build pipelines
        if is_background_task and await is_pipeline_active():
            logger.info("Pipeline active: Silencing background task to save VRAM.")
            return _silent_response(is_native)

    # =============================================================================

    # 2. Fast Exit for Background Traffic
    if not is_streaming and (time.time() < expert_warm_until or vram_locked):
        return _silent_response(is_native, "Analyzing...")

    # 3. Early Command Handling (Before Lock)
    prompt_lower = (messages[-1].get("content", "") if messages else "").lower()
    if not is_background_task:
        if "!lock" in prompt_lower:
            vram_locked = True
            return _command_response("🔒 **VRAM Locked.** Expert model is persistent.", is_streaming, is_native)
        elif "!unlock" in prompt_lower:
            vram_locked = False
            expert_warm_until = 0
            await verified_unload(EXPERT_MODEL)
            await verified_unload(ROUTER_MODEL)
            await free_comfyui()
            return _command_response("🔓 **VRAM Unlocked.** Memory cleared (Expert, Router, & ComfyUI).", is_streaming, is_native)
        elif "!code" in prompt_lower:
            expert_mode = "coding"
            logger.info("Command: Switched to Coding Mode.")
        elif "!general" in prompt_lower:
            expert_mode = "general"
            logger.info("Command: Switched to General Mode.")
        elif "!move" in prompt_lower:
            logger.info("Command: Move initiated.")
            success = mover.handle_move(messages)
            msg = "Files moved!" if "Moved" in success else "Files failed to move!"
            return _command_response(msg, is_streaming, is_native)
        elif "!build" in prompt_lower:
            logger.info("Command: Build pipeline triggered.")
            build_msg = await _trigger_build_pipeline_safe(messages)
            return _command_response(build_msg, is_streaming, is_native)
        elif prompt_lower.startswith("!clone"):
            logger.info("Command: Clone triggered.")
            clone_msg = await asyncio.to_thread(_handle_clone_command, messages)
            return _command_response(clone_msg, is_streaming, is_native)
        elif "!stop" in prompt_lower:
            logger.info("Command: Stop pipeline triggered.")
            stop_msg = _stop_build_pipeline()
            return _command_response(stop_msg, is_streaming, is_native)
        elif "!status" in prompt_lower:
            logger.info("Command: Status check triggered.")
            status_msg = _check_build_status()
            return _command_response(status_msg, is_streaming, is_native)
        elif "!logs" in prompt_lower:
            logger.info("Command: Logs check triggered.")
            logs_msg = _check_build_logs()
            return _command_response(logs_msg, is_streaming, is_native)

    # 4. Request Orchestration
    try:
        # Increase timeout to 5 minutes to accommodate 35B model context processing
        await asyncio.wait_for(gpu_lock.acquire(), timeout=300.0)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=503, content={"error": "The orchestrator is busy."})

    lock_held = True
    try:
        current_time = time.time()
        is_expert_warm = current_time < expert_warm_until

        # Determine target model and load strategy
        target_model = ROUTER_MODEL
        keep_alive = 0
        is_cold_expert = False

        # --- Router-Forwarded Request (from Raspberry Pi) ---
        # If the Pi router already triaged this request and decided it needs
        # the Expert, skip all local triage and go straight to the Expert model.
        is_router_forwarded = request.headers.get("X-Forwarded-By-Router", "").lower() == "true"

        if is_router_forwarded:
            target_model = EXPERT_MODEL
            keep_alive = "10m" if not vram_locked else -1
            # Only treat as cold if the expert isn't already warm and running
            is_cold_expert = not is_expert_warm
            if not vram_locked:
                expert_warm_until = current_time + 600
            logger.info(f"Router-forwarded request: Skipping triage, routing directly to Expert model (cold={is_cold_expert}).")

        # --- Context-Aware Triage Override ---
        elif _get_bound_project_dir(messages) or "@" in prompt_lower:
            project_dir = _get_bound_project_dir(messages)
            target_model = EXPERT_MODEL
            keep_alive = "10m" if not vram_locked else -1
            is_cold_expert = not is_expert_warm
            if not vram_locked:
                expert_warm_until = current_time + 600
            expert_mode = "coding"
            logger.info(f"Project context detected ({'bound' if project_dir else '@mention'}): Routing to Expert model.")
        elif is_background_task or is_image_query or is_search_query or is_search_tool or has_search_history:
            target_model = ROUTER_MODEL
            if is_image_query:
                logger.info("Image generation intent detected: Routing to Router model.")
            elif is_search_query or is_search_tool or has_search_history:
                source = "intent" if is_search_query else "result" if is_search_tool else "history"
                logger.info(f"Search {source} detected: Routing to Router model.")
            else:
                logger.info("Routing background task to Router model.")
        elif any(kw in prompt_lower for kw in ["!bob", "hey bob"]):
            target_model = ROUTER_MODEL
            expert_warm_until = 0
            logger.info("Direct request for Router model.")
        elif any(kw in prompt_lower for kw in ["!expert", "hey expert", "!code", "!general"]):
            target_model = EXPERT_MODEL
            keep_alive = "10m"
            expert_warm_until = current_time + 600
            is_cold_expert = not is_expert_warm
            logger.info(f"Direct request for Expert model ({expert_mode}).")
        elif is_expert_warm or vram_locked:
            target_model = EXPERT_MODEL
            keep_alive = "10m" if not vram_locked else -1
            if not vram_locked:
                expert_warm_until = current_time + 600
        else:
            # Complexity Triage
            await sweep_vram_for_expert()
            analysis = await analyze_request(messages)
            await verified_unload(ROUTER_MODEL)

            if analysis.get("complexity", 1) > 6 or analysis.get("requires_tool", False):
                target_model = EXPERT_MODEL
                keep_alive = "3m" if analysis.get("followups") else 0
                if analysis.get("followups"):
                    expert_warm_until = current_time + 300
                is_cold_expert = True
                expert_mode = "coding" if analysis.get("is_coding") else "general"
                logger.info("Triage: Router model sufficient.")

        # --- Agent Detection ---
        # Detect if this is an automated agent (Cline or Distillation pass)
        is_agent_request = False
        for m in messages:
            if m.get("role") == "system":
                content = str(m.get("content", "")).lower()
                if "you are cline" in content or "distillation" in content or "architect" in content or "engineer" in content:
                    is_agent_request = True
                    break

        # --- Context Binding & Tool Injection ---
        project_dir = _get_bound_project_dir(messages)
        project_context = ""

        # Only inject if it's a bound project AND it's a direct user interaction (not an agent)
        if project_dir and target_model == EXPERT_MODEL and not is_agent_request:
            logger.info(f"Injecting project context from {project_dir}")
            tree = _get_project_tree(project_dir)
            skeleton = _get_symbol_skeleton(project_dir)
            mentions = _parse_file_mentions(messages[-1].get("content", ""), project_dir)

            project_context = (
                f"\n\n<PROJECT_CONTEXT>\n"
                f"You are working on a bound project: `{os.path.basename(project_dir)}`\n"
                f"File Tree:\n{tree}\n"
                f"Symbol Skeleton:\n{skeleton}\n"
            )
            if mentions:
                project_context += mentions
            project_context += "\n</PROJECT_CONTEXT>\n"

            # Inject into the system message (or first available message)
            injected = False
            for m in messages:
                if m.get("role") == "system":
                    m["content"] = str(m.get("content", "")) + project_context
                    injected = True
                    break
            if not injected:
                messages.insert(0, {"role": "system", "content": f"Project Awareness Active.{project_context}"})

        # Define native tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "orchestrator_read_file",
                    "description": "Reads the entire content of a file from the project directory. Use this when the Symbol Skeleton is insufficient.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path to the file"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "orchestrator_expand_dir",
                    "description": "Lists all files and symbols in a specific directory. Use this for recursive exploration of large codebases.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path to the directory"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "orchestrator_build_logs",
                    "description": "Fetches the latest terminal logs from the active background build pipeline (cline-builder Docker container). Use this when the user asks for the build status, errors, or insights into the build process.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ] if project_dir and target_model == EXPERT_MODEL and not is_agent_request else None

        if target_model == EXPERT_MODEL and is_cold_expert:
            await sweep_vram_for_expert()

        # --- Context Pruning (Analytical Resilience) ---
        messages = _prune_messages(messages)
        body["messages"] = messages

        # --- Resolve Provider Config ---
        target_config = _resolve_config(target_model)
        target_is_ollama = _is_ollama_provider(target_config)

        # Pre-load non-Ollama providers (Ollama auto-loads on request)
        if not target_is_ollama and (is_cold_expert or target_model != EXPERT_MODEL):
            await _provider_load(target_config)

        # --- Payload Config ---
        options = body.get("options", {})
        if is_background_task:
            options.update({"temperature": 0.0, "num_ctx": 2048})
            # If the expert is warm or a build is active, force the Router to unload immediately
            if expert_warm_until > time.time() or vram_locked:
                keep_alive = 0
                logger.info("Background task detected in build context: Forcing immediate Router unload.")
        elif target_model == EXPERT_MODEL:
            # Apply Thinking Mode parameters for Expert only if using the default model
            if EXPERT_MODEL == DEFAULT_EXPERT_MODEL:
                params = PARAMS_CODING if expert_mode == "coding" else PARAMS_GENERAL
                options.update(params)
                logger.debug(f"Applied expert parameters for {EXPERT_MODEL}")
            else:
                logger.info(f"Non-default expert {EXPERT_MODEL} detected; using default model settings.")

            options["num_ctx"] = EXPERT_CTX
        else:
            options.update({"temperature": options.get("temperature", 0.7), "num_ctx": EXPERT_CTX})

        body.update({"model": target_model, "keep_alive": keep_alive, "options": options})
        if tools:
            body["tools"] = tools
        logger.info(f"Executing {target_model} (ctx: {options['num_ctx']}, warm: {not is_cold_expert}, tools: {bool(tools)}, provider: {target_config.get('provider', 'ollama')})")

        # --- Compute Target URL ---
        if target_is_ollama:
            target_url = f"{_get_base_url(target_config)}{'/api/chat' if is_native else '/v1/chat/completions'}"
        else:
            target_url = f"{_get_base_url(target_config)}/v1/chat/completions"

        # --- Adapt body for provider ---
        dispatch_body = _adapt_body(body, target_config)

        if tools:
            # Shift to the Agentic Loop handler
            # This will release the lock internally when finished
            try:
                return await _handle_agentic_request(dispatch_body, project_dir, target_url, is_native, gpu_lock, target_is_ollama)
            finally:
                if lock_held:
                    gpu_lock.release()
                    lock_held = False

        if not is_streaming:
            try:
                resp = await http_client.post(target_url, json=dispatch_body, timeout=600.0)
                if resp.status_code != 200:
                    error_text = ""
                    try:
                        error_text = resp.text[:300]
                    except (httpx.HTTPError, UnicodeDecodeError):
                        pass
                    logger.error(f"Inference failed ({resp.status_code}): {error_text}")
                    return JSONResponse(status_code=resp.status_code, content={"error": f"Inference failed: {error_text or 'unknown error'}"})
                data = resp.json()

                # Translate response format if needed
                if not target_is_ollama and is_native:
                    # OpenAI response → Ollama native format
                    choice = data.get("choices", [{}])[0]
                    data = {
                        "model": "Bob",
                        "message": choice.get("message", {"role": "assistant", "content": ""}),
                        "done": True
                    }
                else:
                    data["model"] = "Bob"
                    if "id" in data:
                        data["id"] = "chatcmpl-Bob"
                return JSONResponse(content=data)
            finally:
                if lock_held:
                    gpu_lock.release()
                    lock_held = False

        lock_held = False
        return StreamingResponse(
            stream_proxy(target_url, dispatch_body, gpu_lock,
                         is_native=is_native, backend_is_ollama=target_is_ollama,
                         request_headers=dict(request.headers)),
            media_type="application/x-ndjson" if is_native else "text/event-stream"
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Orchestration Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal orchestration error."})
    finally:
        if lock_held:
            gpu_lock.release()


def _silent_response(is_native: bool, text: str = ""):
    """Returns an empty assistant response to silently terminate an interaction."""
    if is_native:
        return JSONResponse(content={"model": "Bob", "message": {"role": "assistant", "content": text}, "done": True})
    return JSONResponse(content={
        "id": "chatcmpl-Bob", "object": "chat.completion", "created": int(time.time()), "model": "Bob",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
    })


def _command_response(text: str, is_streaming: bool = False, is_native: bool = False):
    """Returns a JSON response (streaming or one-shot) for internal orchestrator commands."""
    if not is_streaming:
        if is_native:
            return JSONResponse(content={"model": "Bob", "message": {"role": "assistant", "content": text}, "done": True})
        return JSONResponse(content={
            "id": "chatcmpl-Bob", "object": "chat.completion", "created": int(time.time()), "model": "Bob",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })

    async def _generator():
        if is_native:
            yield f"{json.dumps({'model': 'Bob', 'message': {'role': 'assistant', 'content': text}, 'done': False})}\n".encode()
            yield f"{json.dumps({'model': 'Bob', 'done': True})}\n".encode()
        else:
            chunk = {
                "id": "chatcmpl-Bob", "object": "chat.completion.chunk", "created": int(time.time()), "model": "Bob",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(_generator(), media_type="application/x-ndjson" if is_native else "text/event-stream")


def _execute_tool(name: str, args: dict, project_dir: str) -> str:
    """Executes a native orchestrator tool."""
    path = args.get("path", "")
    if not path or not project_dir:
        return "Error: Missing path or project directory."

    # Sanitize path
    safe_rel_path = mover.sanitize_path(path)
    abs_path = os.path.join(project_dir, safe_rel_path)

    if name == "orchestrator_read_file":
        try:
            if not os.path.exists(abs_path):
                return f"Error: File `{safe_rel_path}` not found."
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
                return content
        except OSError as e:
            return f"Error reading file: {e}"

    elif name == "orchestrator_build_logs":
        try:
            cmd = "docker ps -a --filter name=cline-builder --format '{{.Names}}' | head -n 1"
            container_name = subprocess.check_output(cmd, shell=True, text=True).strip()
            if not container_name:
                return "Error: No build pipelines found."

            logs_cmd = ["docker", "logs", "--tail", "200", container_name]
            result = subprocess.run(logs_cmd, capture_output=True, text=True, check=False)
            logs_output = (result.stdout + "\n" + result.stderr).strip()

            if not logs_output:
                return f"Logs for {container_name} are currently empty."

            if len(logs_output) > 5000:
                logs_output = "... [Logs Truncated]\n" + logs_output[-5000:]
            return f"Logs for {container_name}:\n{logs_output}"
        except (subprocess.SubprocessError, OSError) as e:
            return f"Error fetching logs: {e}"

    elif name == "orchestrator_expand_dir":
        try:
            if not os.path.exists(abs_path) or not os.path.isdir(abs_path):
                return f"Error: Directory `{safe_rel_path}` not found."

            # List files and symbols for this dir
            skeleton = [f"[EXPANDED DIR: {safe_rel_path}]"]
            signature_re = re.compile(r"^\s*(?:class|def|function|interface|type|async\s+function)\s+([a-zA-Z0-9_]+)", re.MULTILINE)

            for item in os.listdir(abs_path):
                item_path = os.path.join(abs_path, item)
                if os.path.isdir(item_path):
                    if item not in ["node_modules", ".git", "venv", ".venv", "__pycache__", "dist", "build"]:
                        skeleton.append(f"{item}/ (directory)")
                elif item.endswith((".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp")):
                    try:
                        with open(item_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            matches = signature_re.findall(content)
                            skeleton.append(f"{item}")
                            for m in matches:
                                skeleton.append(f"  - {m}")
                    except (OSError, UnicodeDecodeError):
                        skeleton.append(f"{item}")
            return "\n".join(skeleton)
        except OSError as e:
            return f"Error expanding directory: {e}"

    return f"Error: Unknown tool `{name}`"


async def _handle_agentic_request(body: dict, project_dir: str, target_url: str,
                                  is_native: bool, lock: asyncio.Lock, backend_is_ollama: bool = True):
    """
    Handles a tool-enabled request by recursively executing tools and re-running the LLM.
    Returns the final response (one-shot or stream).
    """
    MAX_HOPS = 4
    current_hops = 0

    original_messages = body.get("messages", [])
    # For response parsing: native Ollama format only when backend is Ollama AND client is native
    response_is_native = backend_is_ollama and is_native

    while current_hops < MAX_HOPS:
        current_hops += 1
        logger.info(f"Agentic Loop: Hop {current_hops}/{MAX_HOPS}")

        # We always do a NON-STREAMING call for tools
        temp_body = body.copy()
        temp_body["stream"] = False

        try:
            resp = await http_client.post(target_url, json=temp_body, timeout=600.0)
            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code, content={"error": "Agentic inference failed."})

            data = resp.json()
            message = data.get("message", {}) if response_is_native else data.get("choices", [{}])[0].get("message", {})

            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                # No more tools, this is the final answer
                if not backend_is_ollama and is_native:
                    # Translate OpenAI response to Ollama native
                    choice = data.get("choices", [{}])[0]
                    data = {
                        "model": "Bob",
                        "message": choice.get("message", {"role": "assistant", "content": ""}),
                        "done": True
                    }
                else:
                    data["model"] = "Bob"
                    if "id" in data:
                        data["id"] = "chatcmpl-Bob"
                return JSONResponse(content=data)

            # Execute tools and append results
            original_messages.append(message)

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                logger.info(f"Executing Tool: {name}({args})")
                result = _execute_tool(name, args, project_dir)

                tool_msg = {
                    "role": "tool",
                    "name": name,
                    "content": result,
                    "tool_call_id": tc.get("id", f"call_{int(time.time())}")
                } if response_is_native else {
                    "role": "tool",
                    "content": result
                }
                original_messages.append(tool_msg)

            body["messages"] = original_messages

        except Exception as e:  # noqa: BLE001
            logger.error(f"Agentic Loop Error: {e}")
            break

    return JSONResponse(status_code=500, content={"error": "Agentic loop exceeded max hops or failed."})


# =============================================================================
# INTERNAL MANAGEMENT API (used by distill.py inside Docker containers)
# =============================================================================

@app.post("/internal/model/load")
async def internal_model_load(request: Request):
    """Load a model for a given provider config. Called by distill.py from inside Docker."""
    try:
        config = await request.json()
        # Translate Docker URLs to host URLs
        if "base_url" in config:
            config["base_url"] = _docker_to_host_url(config["base_url"])
        async with gpu_lock:
            await _provider_load(config)
        return JSONResponse(content={"status": "ok"})
    except Exception as e:  # noqa: BLE001
        logger.error(f"Internal model load failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/internal/model/unload")
async def internal_model_unload(request: Request):
    """Unload a model for a given provider config. Called by distill.py from inside Docker."""
    try:
        config = await request.json()
        if "base_url" in config:
            config["base_url"] = _docker_to_host_url(config["base_url"])
        model_name = config.get("model", "")
        provider = config.get("provider", "ollama")

        if provider == "ollama":
            base_url = config.get("base_url", "http://localhost:11434")
            try:
                await http_client.post(
                    f"{base_url}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=5.0
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Internal Ollama unload failed: {e}")
        elif provider == "lmstudio":
            proc = await asyncio.create_subprocess_exec(
                "lms", "unload", model_name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
        elif provider == "llamacpp":
            await _stop_llamacpp_server(config)

        return JSONResponse(content={"status": "ok"})
    except Exception as e:  # noqa: BLE001
        logger.error(f"Internal model unload failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


def _get_bound_project_dir(messages: list) -> str | None:
    """Finds the project directory bound to the current conversation ID."""
    conv_id = mover.get_conversation_id(messages)
    conv_dir = "conversations"
    if not os.path.exists(conv_dir):
        return None

    for item in os.listdir(conv_dir):
        if item.endswith(f"_{conv_id}"):
            return os.path.join(conv_dir, item)
    return None


def _get_project_tree(project_dir: str) -> str:
    """Generates a pruned directory tree for the project."""
    try:
        # Full tree but pruned of noise
        cmd = ["tree", project_dir, "-I", "node_modules|.git|venv|.venv|__pycache__|dist|build|public|.knowledge_base|.cline_context|.cline_logs"]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)

        # Scale guard: truncate if too large
        if len(output) > 5000:
            # Fallback to depth 1 if still too large? Or just truncate.
            return output[:5000] + "\n... [Tree truncated for context safety]"
        return output
    except (subprocess.SubprocessError, OSError):
        return ""


def _get_symbol_skeleton(project_dir: str) -> str:
    """Matches class and function signatures to create a project skeleton."""
    skeleton = ["[PROJECT SYMBOL SKELETON]"]

    # Simple regex-based extraction for Python, JS/TS, etc.
    # class\s+(\w+)|function\s+(\w+)|(\w+)\s*=\s*(async\s*)?\([^)]*\)\s*=>|def\s+(\w+)
    signature_re = re.compile(r"^\s*(?:class|def|function|interface|type|async\s+function)\s+([a-zA-Z0-9_]+)", re.MULTILINE)

    total_chars = 0
    MAX_SKELETON_CHARS = 10000

    for root, dirs, files in os.walk(project_dir):
        # Prune dirs
        dirs[:] = [d for d in dirs if d not in ["node_modules", ".git", "venv", ".venv", "__pycache__", "dist", "build", "public", ".knowledge_base", ".cline_context", ".cline_logs"]]

        for file in files:
            if file.endswith((".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp")):
                rel_path = os.path.relpath(os.path.join(root, file), project_dir)
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()
                        matches = signature_re.findall(content)
                        if matches:
                            file_block = [f"\n{rel_path}"]
                            for m in matches:
                                file_block.append(f"  - {m}")

                            block_str = "\n".join(file_block)
                            if total_chars + len(block_str) > MAX_SKELETON_CHARS:
                                skeleton.append("\n... [Skeleton truncated: Use <expand_dir> or @filename for more detail]")
                                return "\n".join(skeleton)

                            skeleton.append(block_str)
                            total_chars += len(block_str)
                except (OSError, UnicodeDecodeError):
                    continue

    return "\n".join(skeleton)


def _parse_file_mentions(text: str, project_dir: str) -> str:
    """Detects @filename mentions and reads their content."""
    mentions = re.findall(r"@([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)", text)
    if not mentions:
        return ""

    context_blocks = ["\n[REQUESTED FILE CONTENT]"]
    for filename in mentions:
        # Search for file in project_dir
        found_path = None
        for root, _, files in os.walk(project_dir):
            if filename in files:
                found_path = os.path.join(root, filename)
                break
            # Check for partial path matches (e.g. @src/main.py)
            normalized_target = filename.replace('\\', '/')
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), project_dir).replace('\\', '/')
                if rel == normalized_target or rel.endswith("/" + normalized_target):
                    found_path = os.path.join(root, f)
                    break
            if found_path:
                break

        if found_path:
            try:
                with open(found_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # No truncation for explicit @mentions as per plan
                    context_blocks.append(f'\n<file name="{os.path.relpath(found_path, project_dir)}">\n{content}\n</file>')
            except OSError as e:
                context_blocks.append(f"\n[Error reading {filename}: {e}]")

    return "\n".join(context_blocks) if len(context_blocks) > 1 else ""


def _handle_clone_command(messages: list) -> str:
    """Handles !clone command by cloning the repo into the conversation project directory."""
    try:
        last_msg = messages[-1].get("content", "").strip()
        parts = last_msg.split()
        repo_url = None
        kb_url = None

        for i, part in enumerate(parts):
            if part.startswith(("http", "git@")):
                if i > 0 and parts[i-1] == "--kb":
                    kb_url = part
                elif repo_url is None:
                    repo_url = part
            elif part == "--repo" and i + 1 < len(parts):
                repo_url = parts[i+1]

        # Resolve target_dir
        repo_url_provided = repo_url is not None
        bound_dir = _get_bound_project_dir(messages)

        target_dir = None
        repo_name = "project"

        if repo_url_provided:
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            prefix = repo_name[:60]
            conv_id = mover.get_conversation_id(messages)
            target_dir = os.path.join("conversations", f"{prefix}_{conv_id}")
        elif bound_dir:
            target_dir = bound_dir
            repo_name = os.path.basename(bound_dir).split('_')[0]
        else:
            return "❌ **Clone Failed:** No repository URL provided and no existing project is bound."

        if os.path.exists(target_dir):
            if not kb_url:
                return f"✅ **Project Bound:** `{repo_name}` is already linked to this conversation."

            # If KB is provided, check if we need to clone it
            kb_dir = os.path.join(target_dir, ".knowledge_base")
            if os.path.exists(kb_dir):
                return f"✅ **Project & KB Bound:** `{repo_name}` is already linked with its knowledge base."

            logger.info(f"Existing project found, but KB is missing. Cloning KB from {kb_url}...")
            subprocess.run(["git", "clone", kb_url, kb_dir], check=True, capture_output=True)
            return f"✅ **Linked Knowledge Base:** Added `.knowledge_base/` to existing project `{repo_name}`."

        subprocess.run(["git", "clone", repo_url, target_dir], check=True, capture_output=True)

        msg = f"✅ **Cloned Repository:** `{repo_name}` into bound conversation folder.\n"

        if kb_url:
            kb_dir = os.path.join(target_dir, ".knowledge_base")
            subprocess.run(["git", "clone", kb_url, kb_dir], check=True, capture_output=True)
            msg += "✅ **Linked Knowledge Base:** Cloned into `.knowledge_base/`.\n"

        # Get pruned directory tree
        tree_cmd = ["tree", target_dir, "-L", "3", "-I", "node_modules|.git|venv|.venv|__pycache__|dist|build|public|.knowledge_base"]
        try:
            tree_output = subprocess.check_output(tree_cmd, text=True, stderr=subprocess.DEVNULL)
            msg += f"\n**Directory Structure:**\n```\n{tree_output}\n```"
        except (subprocess.SubprocessError, OSError):
            pass

        return msg

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else e}")
        return "❌ **Clone Failed:** Process error occurred."
    except Exception as e:  # noqa: BLE001
        return f"❌ **Error during clone:** {e}"


async def _trigger_build_pipeline(messages: list) -> str:
    """
    Orchestrates the build pipeline:
    1. Extracts conversation files to a unique project folder.
    2. Serializes the conversation for distillation.
    3. Launches the cline-builder Docker container.
    """
    try:
        # 1. Handle direct cloning if repo info is provided in the build command
        last_msg = messages[-1].get("content", "").strip()
        if "--repo" in last_msg or "http" in last_msg or "git@" in last_msg:
            logger.info("Direct repo/kb info detected in !build command. Triggering clone sync...")
            clone_status = _handle_clone_command(messages)
            if "❌" in clone_status:
                return clone_status

        # 2. Extract files via mover
        # This creates the project folder in ./conversations/
        status_msg = mover.handle_move(messages)
        target_dir = None
        if "No code snippets" in status_msg:
            # Check if we already have a bound project
            bound_dir = _get_bound_project_dir(messages)
            if bound_dir:
                logger.info(f"No snippets found, but project is bound to {bound_dir}. Proceeding with build.")
                target_dir = bound_dir
            else:
                # No code snippets and no bound project — create a conversation-only folder
                # The user may be discussing requirements without code yet
                conv_id = mover.get_conversation_id(messages)
                target_dir = os.path.join("conversations", f"project_{conv_id}")
                os.makedirs(target_dir, exist_ok=True)
                logger.info(f"No snippets or bound project. Created conversation-only folder: {target_dir}")
        else:
            # Extract target_dir from the mover's response
            match = re.search(r"to `([^`]+)`", status_msg)
            if not match:
                # Check if mover returned an explicit error
                if "Error" in status_msg:
                    return f"❌ **Build Aborted.** {status_msg}"
                return f"❌ **Critical Error:** Failed to determine workspace path. {status_msg}"
            target_dir = match.group(1)

        # We already have target_dir from above
        abs_target_dir = os.path.abspath(target_dir)

        # Create context directory
        context_dir = os.path.join(abs_target_dir, ".cline_context")
        os.makedirs(context_dir, exist_ok=True)
        conv_file_path = os.path.join(context_dir, "conversation.json")

        # 2. Save conversation for the container
        with open(conv_file_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            json.dump(messages, f, indent=2)

        # 3. Launch Docker Container (Non-blocking)
        # We mount the SPECIFIC conversation folder as /workspace
        container_name = f"cline-builder-{int(time.time())}"
        project_name = os.path.basename(abs_target_dir)
        cmd = [
            "docker", "compose", "--profile", "build", "run", "-d", "--rm",
            "--name", container_name,
            "-v", f"{abs_target_dir}:/workspace",
            "-e", "CONVERSATION_FILE=/workspace/.cline_context/conversation.json",
            "-e", f"PROJECT_NAME={project_name}",
            "-e", f"EXPERT_CTX={DISTILL_CTX}",
            "-e", f"CLINE_CTX={CLINE_CTX}",
            "-e", "ORCHESTRATOR_URL=http://host.docker.internal:8000",
            "cline-builder"
        ]

        logger.info(f"Launching pipeline command: {' '.join(cmd)}")
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: ASYNC220

        # Launch the Safety Monitor to ensure VRAM is released when container stops/deleted
        asyncio.create_task(_docker_safety_monitor(container_name))

        return (
            f"🔨 **Build pipeline triggered.**\n\n"
            f"- **Workspace:** `conversations/{os.path.basename(target_dir)}`\n"
            f"- **Container:** `{container_name}`\n"
            f"- **Mode:** `Autonomous`\n"
            f"- **Status:** *Distillation started... Cline will follow.*\n\n"
            f"Check logs with: `docker logs -f {container_name}`\n"
            f"Tip: You can also type `!logs` to view them in chat, or ask me for build insights!"
        )

    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to trigger build: {e}")
        return f"❌ **Build Trigger Failed:** {e}"


def _check_build_logs() -> str:
    """Fetches the latest logs from the active or most recent build pipeline."""
    try:
        cmd = "docker ps -a --filter name=cline-builder --format '{{.Names}}' | head -n 1"
        container_name = subprocess.check_output(cmd, shell=True, text=True).strip()

        if not container_name:
            return "📭 **No build pipelines found.**"

        logs_cmd = ["docker", "logs", "--tail", "50", container_name]
        result = subprocess.run(logs_cmd, capture_output=True, text=True, check=False)

        logs_output = (result.stdout + "\n" + result.stderr).strip()

        if not logs_output:
            logs_output = "[No logs generated yet or container is empty]"

        if len(logs_output) > 3000:
            logs_output = "... [truncated]\n" + logs_output[-3000:]

        return f"📜 **Latest logs for `{container_name}`:**\n```text\n{logs_output}\n```"
    except (subprocess.SubprocessError, OSError) as e:
        return f"❌ **Failed to fetch logs:** {e}"


def _check_build_status() -> str:
    """Checks for active or recently completed cline-builder containers."""
    try:
        cmd = ["docker", "ps", "-a", "--filter", "name=cline-builder", "--format", "{{.Names}}\t{{.Status}}\t{{.RunningFor}}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if not result.stdout.strip():
            return "📭 **No active or recent build pipelines found.**"

        lines = result.stdout.strip().split("\n")
        report = ["🔍 **Build Pipeline Status:**", ""]
        for line in lines:
            name, status, age = line.split("\t")
            icon = "🟢" if "Up" in status else "⚪"
            report.append(f"{icon} `{name}`: {status} (Started {age} ago)")

        return "\n".join(report)
    except (subprocess.SubprocessError, OSError) as e:
        return f"❌ **Status Check Failed:** {e}"


async def is_pipeline_active() -> bool:
    """Checks if any cline-builder container is currently running."""
    try:
        # Run docker ps to see if any containers with the name 'cline-builder' are running
        cmd = ["docker", "ps", "--filter", "name=cline-builder", "--filter", "status=running", "--format", "{{.ID}}"]
        # Use asyncio.create_subprocess_exec for non-blocking check
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        return bool(stdout.strip())
    except (asyncio.SubprocessError, OSError) as e:
        logger.warning(f"Failed to check pipeline activity: {e}")
        return False


async def _docker_safety_monitor(container_name: str):
    """
    Background task that monitors a specific Docker container.
    When the container stops, it automatically triggers a VRAM cleanup.
    """
    logger.info(f"[Safety Monitor] Starting monitor for {container_name}")
    try:
        # Initial wait to let it start
        await asyncio.sleep(10)

        while True:
            # Check container status
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f", "{{.State.Running}}", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _stderr = await proc.communicate()

            if proc.returncode != 0:
                # Container likely deleted or vanished
                logger.info(f"[Safety Monitor] Container {container_name} vanished. Releasing VRAM.")
                await shutdown_expert()
                break

            status = stdout.decode().strip().lower()
            if status == "false":
                logger.info(f"[Safety Monitor] Container {container_name} stopped. Releasing VRAM.")
                await shutdown_expert()
                break

            # Wait before next poll
            await asyncio.sleep(10)

    except asyncio.CancelledError:
        logger.info(f"[Watchdog] Monitor for {container_name} cancelled.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"[Watchdog] Error monitoring {container_name}: {e}")


async def _trigger_build_pipeline_safe(messages: list) -> str:
    """
    Triggers the build pipeline while ensuring VRAM is cleared and
    waiting for any active LLM tasks to finish.
    Includes a cooldown to prevent duplicate triggers from client retries.
    """
    global _last_build_trigger_time, vram_locked
    current_time = time.time()
    if current_time - _last_build_trigger_time < 10:
        logger.warning("Duplicate build trigger ignored (cooldown).")
        return "⚠️ **Build already starting.** Please wait a few seconds for the status update."

    _last_build_trigger_time = current_time

    async with gpu_lock:
        logger.info("Acquired GPU lock for pipeline trigger. Clearing VRAM...")
        # Protect VRAM from periodic cleanup while build runs
        vram_locked = True

        # Don't kill the Expert if it is a managed llama.cpp process
        # This prevents the 'Network unreachable' error when the container starts
        expert_cfg = _resolve_config(EXPERT_MODEL)
        if expert_cfg.get("provider") != "llamacpp":
            await verified_unload(EXPERT_MODEL)

        # Router and ComfyUI should always be cleared as they are small/fast to reload
        await verified_unload(ROUTER_MODEL)
        await free_comfyui()

        # Now trigger the actual command
        return await _trigger_build_pipeline(messages)


def _stop_build_pipeline() -> str:
    """Stops and removes all cline-builder containers."""
    try:
        # Get list of containers
        list_cmd = ["docker", "ps", "-a", "--filter", "name=cline-builder", "--format", "{{.Names}}"]
        containers = subprocess.run(list_cmd, capture_output=True, text=True, check=False).stdout.strip().split("\n")

        if not containers or not containers[0]:
            return "📭 **No active build pipelines to stop.**"

        # Stop and remove them
        for name in containers:
            if name:
                logger.info(f"Stopping container: {name}")
                subprocess.run(["docker", "stop", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                subprocess.run(["docker", "rm", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        return f"🛑 **Stopped and cleared {len(containers)} build pipeline(s).**"
    except (subprocess.SubprocessError, OSError) as e:
        return f"❌ **Stop Command Failed:** {e}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="warning")
