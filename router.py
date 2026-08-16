import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Load environment from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Bob-Router")

# =============================================================================
# CONFIGURATION
# =============================================================================

ROUTER_PORT = int(os.getenv("ROUTER_PORT", "8001"))
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen2.5:1.5b")
ROUTER_OLLAMA_URL = os.getenv("ROUTER_OLLAMA_URL", "http://localhost:11434")
ROUTER_CTX = int(os.getenv("ROUTER_CTX", "4096"))

# Desktop (Expert Node)
DESKTOP_IP = os.getenv("DESKTOP_IP", "192.168.100.164")
DESKTOP_PORT = int(os.getenv("DESKTOP_PORT", "8000"))
DESKTOP_ORCHESTRATOR_URL = f"http://{DESKTOP_IP}:{DESKTOP_PORT}"

# Desktop Waker Service (already running on this Pi at port 8000)
WAKER_URL = os.getenv("WAKER_URL", "http://localhost:8000")
WAKER_TOKEN = os.getenv("WAKER_TOKEN", "")

# Desktop remote startup (SSH into desktop to launch orchestrator on demand)
DESKTOP_SSH_USER = os.getenv("DESKTOP_SSH_USER", "mv")
DESKTOP_SSH_HOST = os.getenv("DESKTOP_SSH_HOST", DESKTOP_IP)
DESKTOP_WORKSPACE_DIR = os.getenv("DESKTOP_WORKSPACE_DIR", "/home/mv/ghub/br.ai.n")

# WoL Timing
WOL_BOOT_WAIT = int(os.getenv("WOL_BOOT_WAIT", "35"))        # Seconds to wait for OS boot after WoL
WOL_HEALTH_TIMEOUT = int(os.getenv("WOL_HEALTH_TIMEOUT", "90"))  # Seconds to poll /health after SSH startup
WOL_POLL_INTERVAL = int(os.getenv("WOL_POLL_INTERVAL", "5"))

# Optional: Kill GUI on desktop to free ~2.5GB VRAM for AI workloads
KILL_GUI_ON_WAKE = os.getenv("KILL_GUI_ON_WAKE", "false").lower() == "true"

# Complexity threshold: queries above this score get forwarded to the desktop expert
COMPLEXITY_THRESHOLD = int(os.getenv("COMPLEXITY_THRESHOLD", "6"))

# =============================================================================
# STATE
# =============================================================================

http_client: httpx.AsyncClient = None
_desktop_online: bool = False
_last_desktop_check: float = 0
_desktop_check_interval = 30  # Cache desktop status for 30s
_expert_warm_until: float = 0  # Mirrors desktop's expert_warm_until — skips triage while warm


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifecycle."""
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    logger.info("Router HTTP client initialized.")

    # Start background desktop health monitor
    monitor_task = asyncio.create_task(_desktop_health_monitor())

    yield

    monitor_task.cancel()
    await http_client.aclose()
    logger.info("Router HTTP client closed.")


app = FastAPI(title="Bob: Router Node (Raspberry Pi)", lifespan=lifespan)


# =============================================================================
# DESKTOP CONNECTIVITY & WAKE-ON-LAN
# =============================================================================

async def _desktop_health_monitor():
    """Background task that periodically checks if the desktop is reachable."""
    global _desktop_online, _last_desktop_check
    while True:
        try:
            _desktop_online = await _check_desktop_health()
            _last_desktop_check = time.time()
        except asyncio.CancelledError:
            break
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Desktop health monitor error: {e}")
            _desktop_online = False
        await asyncio.sleep(_desktop_check_interval)


async def _check_desktop_health() -> bool:
    """Quick health check against the desktop orchestrator's /health endpoint."""
    try:
        resp = await http_client.get(f"{DESKTOP_ORCHESTRATOR_URL}/health", timeout=3.0)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


async def is_desktop_online() -> bool:
    """Returns cached desktop status, refreshing if stale."""
    global _desktop_online, _last_desktop_check
    if time.time() - _last_desktop_check > _desktop_check_interval:
        _desktop_online = await _check_desktop_health()
        _last_desktop_check = time.time()
    return _desktop_online


async def _send_wol() -> bool:
    """
    Sends Wake-on-LAN via the existing desktop-waker service on this Pi.
    The waker service (main.py) is already running on port 8000 and handles
    the actual wakeonlan CLI call + MAC address.
    """
    if not WAKER_TOKEN:
        logger.error("WoL failed: WAKER_TOKEN not configured in .env")
        return False
    try:
        resp = await http_client.get(
            f"{WAKER_URL}/wake-desktop",
            headers={"x-auth-token": WAKER_TOKEN},
            timeout=5.0
        )
        if resp.status_code == 200:
            logger.info("WoL magic packet sent via desktop-waker service.")
            return True
        else:
            body = resp.text[:200] if hasattr(resp, 'text') else str(resp.status_code)
            logger.error(f"Waker service returned {resp.status_code}: {body}")
            return False
    except httpx.ConnectError:
        logger.error(f"Waker service unreachable at {WAKER_URL}. Is desktop-waker.service running?")
        return False
    except Exception as e:  # noqa: BLE001
        logger.error(f"WoL request failed: {e}")
        return False


async def _kill_desktop_gui():
    """
    Calls the waker service's /kill-gui endpoint to stop the display manager.
    Frees ~2.5GB VRAM on the desktop for AI workloads.
    """
    if not KILL_GUI_ON_WAKE:
        return
    try:
        resp = await http_client.get(
            f"{WAKER_URL}/kill-gui",
            headers={"x-auth-token": WAKER_TOKEN},
            timeout=15.0
        )
        if resp.status_code == 200:
            logger.info("Desktop GUI stopped to free VRAM.")
        else:
            logger.warning(f"Kill GUI returned {resp.status_code} (may already be stopped).")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Kill GUI failed (non-critical): {e}")


async def _start_desktop_services():
    """
    SSH into the desktop and launch the orchestrator + ComfyUI.
    Uses start_desktop.sh which starts services in the background via nohup.
    """
    cmd = (
        f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes "
        f"{DESKTOP_SSH_USER}@{DESKTOP_SSH_HOST} "
        f"'cd {DESKTOP_WORKSPACE_DIR} && bash start_desktop.sh'"
    )
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode == 0:
            logger.info("Desktop services started via SSH.")
            if stdout:
                logger.info(f"SSH stdout: {stdout.decode().strip()}")
        else:
            logger.warning(f"SSH startup returned code {proc.returncode}: {stderr.decode().strip()}")
    except asyncio.TimeoutError:
        logger.warning("SSH startup timed out (services may still be starting).")
    except Exception as e:  # noqa: BLE001
        logger.error(f"SSH startup failed: {e}")


async def ensure_desktop_ready() -> bool:
    """
    Full desktop wake + start sequence:
    1. Check if already online (skip everything if so)
    2. Send WoL via waker service
    3. Wait for OS to boot (~35s)
    4. Kill GUI to free VRAM (optional)
    5. SSH into desktop to start orchestrator + ComfyUI
    6. Poll /health until the orchestrator is responding
    """
    global _desktop_online, _last_desktop_check

    # 1. Quick check — maybe it's already running
    if await _check_desktop_health():
        _desktop_online = True
        _last_desktop_check = time.time()
        return True

    # 2. Send WoL
    logger.info("Desktop is offline. Sending Wake-on-LAN...")
    if not await _send_wol():
        return False

    # 3. Wait for the OS to boot
    logger.info(f"Waiting {WOL_BOOT_WAIT}s for desktop to boot...")
    await asyncio.sleep(WOL_BOOT_WAIT)

    # 4. Kill GUI to free VRAM (before starting heavy AI services)
    await _kill_desktop_gui()

    # 5. Start orchestrator + ComfyUI via SSH
    logger.info("Starting desktop services via SSH...")
    await _start_desktop_services()

    # 6. Poll /health until the orchestrator is ready
    logger.info(f"Polling desktop /health for up to {WOL_HEALTH_TIMEOUT}s...")
    deadline = time.time() + WOL_HEALTH_TIMEOUT
    wol_resend_time = time.time()

    while time.time() < deadline:
        await asyncio.sleep(WOL_POLL_INTERVAL)

        if await _check_desktop_health():
            logger.info("Desktop orchestrator is online and ready!")
            _desktop_online = True
            _last_desktop_check = time.time()
            return True

        # Resend WoL every 30s in case the desktop didn't wake properly
        if time.time() - wol_resend_time > 30:
            logger.info("Resending WoL packet (safety retry)...")
            await _send_wol()
            wol_resend_time = time.time()

    logger.error(f"Desktop orchestrator did not become ready within {WOL_HEALTH_TIMEOUT}s.")
    return False


# =============================================================================
# TRIAGE (Local, on the Pi's Qwen 2.5 1.5B)
# =============================================================================

async def analyze_request(messages: list) -> dict:
    """
    Lightweight triage using the local Qwen 2.5 1.5B model.
    Determines if a request needs the desktop expert or can be handled locally.
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
        "Also determine if this is a 'coding' task. "
        'Respond ONLY in pure JSON format: {"complexity": <int>, "expect_followups": <bool>, "is_coding": <bool>}'
    )

    payload = {
        "model": ROUTER_MODEL,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": context_text[:1000]}
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0.0, "num_ctx": 2048}
    }

    try:
        resp = await http_client.post(
            f"{ROUTER_OLLAMA_URL}/api/chat", json=payload, timeout=15.0
        )
        if resp.status_code == 200:
            content = resp.json().get("message", {}).get("content", "{}")
            data = json.loads(content)
            return {
                "complexity": int(data.get("complexity", 1)),
                "followups": bool(data.get("expect_followups", False)),
                "is_coding": bool(data.get("is_coding", False))
            }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Triage error: {e}")

    return {"complexity": 1, "followups": False, "is_coding": False}


# =============================================================================
# REQUEST CLASSIFICATION (Fast, no LLM needed)
# =============================================================================

def _classify_request(messages: list) -> str:
    """
    Fast pre-classification of a request. Runs BEFORE the LLM-based triage.
    Returns one of: 'desktop', 'local', 'background', 'triage_needed'.
    """
    if not messages:
        return "local"

    last_msg = messages[-1]
    last_content = str(last_msg.get("content", "")).lower()
    role = last_msg.get("role", "")

    # --- Desktop Commands (always forward) ---
    desktop_commands = [
        "!build", "!move", "!lock", "!unlock", "!stop", "!status",
        "!logs", "!clone", "!expert", "!code", "!general",
        "hey expert", "hey code"
    ]
    if any(cmd in last_content for cmd in desktop_commands):
        return "desktop"

    # --- Agent Detection (Cline, Distillation system prompts) ---
    for m in messages:
        if m.get("role") == "system":
            sys_content = str(m.get("content", "")).lower()
            if any(kw in sys_content for kw in ["you are cline", "distillation", "architect", "engineer"]):
                return "desktop"

    # --- Project Context (@file mentions) ---
    if "@" in last_content and role == "user" and re.search(r"@[a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+", last_content):
        return "desktop"

    # --- Image Generation Intent ---
    image_triggers = [
        "generate an image", "create an image", "create a picture",
        "draw a", "make an image", "flux", "comfyui"
    ]
    if role == "user" and any(kw in last_content for kw in image_triggers):
        return "desktop"

    # --- Background Tasks (handle locally, quick and cheap) ---
    bg_keywords = [
        "suggest 3-5", "generate a title", "generate a short title",
        "summarize", "short label", "tags"
    ]
    if any(kw in last_content for kw in bg_keywords):
        return "background"

    # --- Explicit local request ---
    if any(kw in last_content for kw in ["!bob", "hey bob"]):
        return "local"

    # --- Everything else needs LLM triage ---
    return "triage_needed"


# =============================================================================
# RESPONSE HELPERS
# =============================================================================

def _silent_response(is_native: bool, text: str = ""):
    """Returns an empty assistant response to silently terminate an interaction."""
    if is_native:
        return JSONResponse(content={
            "model": "Bob", "message": {"role": "assistant", "content": text}, "done": True
        })
    return JSONResponse(content={
        "id": "chatcmpl-Bob", "object": "chat.completion",
        "created": int(time.time()), "model": "Bob",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
    })


def _command_response(text: str, is_streaming: bool = False, is_native: bool = False):
    """Returns a formatted response for command/status messages."""
    if not is_streaming:
        if is_native:
            return JSONResponse(content={
                "model": "Bob", "message": {"role": "assistant", "content": text}, "done": True
            })
        return JSONResponse(content={
            "id": "chatcmpl-Bob", "object": "chat.completion",
            "created": int(time.time()), "model": "Bob",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })

    async def _generator():
        if is_native:
            yield f"{json.dumps({'model': 'Bob', 'message': {'role': 'assistant', 'content': text}, 'done': False})}\n".encode()
            yield f"{json.dumps({'model': 'Bob', 'done': True})}\n".encode()
        else:
            chunk = {
                "id": "chatcmpl-Bob", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": "Bob",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        _generator(),
        media_type="application/x-ndjson" if is_native else "text/event-stream"
    )


def _error_chunk(msg: str, is_native: bool) -> bytes:
    """Generates a single error chunk in the appropriate format."""
    if is_native:
        return f'{json.dumps({"model": "Bob", "message": {"role": "assistant", "content": f"⚠️ {msg}"}, "done": True})}\n'.encode()
    chunk = {
        "id": "chatcmpl-Bob", "object": "chat.completion.chunk", "model": "Bob",
        "choices": [{"index": 0, "delta": {"content": f"⚠️ {msg}"}, "finish_reason": "stop"}]
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


# =============================================================================
# STREAM PROXYING
# =============================================================================

async def _proxy_stream_local(url: str, body: dict, is_native: bool):
    """
    Stream proxy to local Ollama on the Pi.
    Minimal processing — just rewrites the model name to 'Bob'.
    No syntax scrubbing needed (1.5B doesn't hallucinate nested XML tags).
    """
    try:
        async with http_client.stream("POST", url, json=body, timeout=600.0) as resp:
            if resp.status_code != 200:
                error_body = ""
                async for chunk in resp.aiter_bytes():
                    error_body += chunk.decode("utf-8", errors="ignore")
                logger.error(f"Local inference error ({resp.status_code}): {error_body[:300]}")
                yield _error_chunk(error_body[:200], is_native)
                return

            async for line in resp.aiter_lines():
                if not line:
                    continue

                if is_native:
                    # Ollama native NDJSON format
                    try:
                        data = json.loads(line)
                        data["model"] = "Bob"
                        yield f"{json.dumps(data)}\n".encode()
                    except json.JSONDecodeError:
                        yield f"{line}\n".encode()
                else:
                    # OpenAI SSE format (data: {json})
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        yield b"data: [DONE]\n\n"
                        continue
                    try:
                        data = json.loads(line[6:])
                        data["model"] = "Bob"
                        if "id" in data:
                            data["id"] = "chatcmpl-Bob"
                        yield f"data: {json.dumps(data)}\n\n".encode()
                    except json.JSONDecodeError:
                        yield f"{line}\n\n".encode()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Local stream error: {e}")
        yield _error_chunk(str(e), is_native)


async def _proxy_stream_desktop(url: str, body: dict, is_native: bool, headers: dict):
    """
    Transparent stream proxy to the desktop orchestrator.
    Zero processing — the desktop already handles model name rewriting and scrubbing.
    """
    try:
        async with http_client.stream("POST", url, json=body, headers=headers, timeout=600.0) as resp:
            if resp.status_code != 200:
                error_body = ""
                async for chunk in resp.aiter_bytes():
                    error_body += chunk.decode("utf-8", errors="ignore")
                logger.error(f"Desktop inference error ({resp.status_code}): {error_body[:300]}")
                yield _error_chunk(f"Desktop error: {error_body[:200]}", is_native)
                return

            # Pure byte passthrough — desktop already handles everything
            async for chunk in resp.aiter_bytes():
                yield chunk
    except httpx.ConnectError:
        logger.error("Desktop connection refused — orchestrator may not be running.")
        yield _error_chunk("Desktop orchestrator is not responding. It may still be starting up.", is_native)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Desktop stream error: {e}")
        yield _error_chunk(f"Desktop unreachable: {e}", is_native)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/tags")
@app.get("/v1/models")
async def list_models(request: Request):
    """Returns the unified model list — Bob is the only model clients see."""
    is_tags = "tags" in str(request.url)
    bob_model = {
        "name": "Bob", "model": "Bob",
        "modified_at": "2026-03-16T00:00:00Z", "size": 0,
        "digest": "bob-identity",
        "details": {"family": "llama", "parameter_size": "Router+Expert", "quantization_level": "Q8_0"}
    } if is_tags else {
        "id": "Bob", "object": "model", "created": int(time.time()), "owned_by": "System"
    }

    if is_tags:
        return JSONResponse(content={"models": [bob_model]})
    return JSONResponse(content={"object": "list", "data": [bob_model]})


@app.get("/health")
async def health_check():
    """Health endpoint showing router status and desktop reachability."""
    desktop_status = await is_desktop_online()
    return JSONResponse(content={
        "node": "router",
        "status": "ok",
        "router_model": ROUTER_MODEL,
        "desktop_online": desktop_status,
        "desktop_url": DESKTOP_ORCHESTRATOR_URL,
    })


@app.get("/wake-desktop")
async def wake_desktop_endpoint():
    """Manual endpoint to wake the desktop and start AI services."""
    success = await ensure_desktop_ready()
    if success:
        return JSONResponse(content={"status": "success", "message": "Desktop is online and ready."})
    return JSONResponse(status_code=503, content={
        "status": "failed",
        "message": "Desktop did not respond. Check WoL and SSH configuration."
    })


# =============================================================================
# MAIN ROUTING ENDPOINT
# =============================================================================

@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def route_request(request: Request):
    """
    Main entry point for all AI chat requests.

    Flow:
    1. Fast classification (keyword-based, no LLM)
    2. If ambiguous, run local LLM triage (Qwen 2.5 1.5B)
    3. Route to either local Ollama or desktop orchestrator
    """
    body = await request.json()
    messages = body.get("messages", [])
    is_streaming = body.get("stream", True)
    path = str(request.url.path)
    is_native = "/api/chat" in path

    # --- Step 1: Fast classification (no LLM needed) ---
    classification = _classify_request(messages)
    logger.info(f"Request classified as: {classification}")

    # --- Step 2: Route based on classification ---

    if classification == "desktop":
        _set_expert_warm()
        return await _forward_to_desktop(body, path, is_native, is_streaming)

    if classification == "background":
        return await _handle_locally(body, path, is_native, is_streaming, keep_alive="1m")

    if classification == "local":
        return await _handle_locally(body, path, is_native, is_streaming, keep_alive="5m")

    # --- Step 2.5: Expert Warm Bypass ---
    # If the desktop expert was recently invoked, skip triage and forward directly.
    # This ensures follow-up questions go to the expert without re-triaging.
    if _is_expert_warm():
        logger.info("Expert is warm — forwarding directly to desktop (skipping triage).")
        return await _forward_to_desktop(body, path, is_native, is_streaming)

    # --- Step 3: LLM-based triage (complexity analysis) ---
    analysis = await analyze_request(messages)
    logger.info(f"Triage result: complexity={analysis['complexity']}, coding={analysis['is_coding']}")

    if analysis.get("complexity", 1) > COMPLEXITY_THRESHOLD:
        logger.info(f"Complexity {analysis['complexity']} > {COMPLEXITY_THRESHOLD}: Forwarding to desktop expert.")
        _set_expert_warm()
        return await _forward_to_desktop(body, path, is_native, is_streaming)

    # Simple enough — handle locally on the Pi
    keep_alive = "5m" if analysis.get("followups") else "2m"
    return await _handle_locally(body, path, is_native, is_streaming, keep_alive=keep_alive)


# =============================================================================
# ROUTING HANDLERS
# =============================================================================

async def _handle_locally(body: dict, path: str, is_native: bool,
                          is_streaming: bool, keep_alive: str = "5m"):
    """Handle a request using the local Ollama model on the Pi."""
    body["model"] = ROUTER_MODEL
    body["keep_alive"] = keep_alive

    # Set context window for local model
    options = body.get("options", {})
    options["num_ctx"] = ROUTER_CTX
    body["options"] = options

    target_url = f"{ROUTER_OLLAMA_URL}{'/api/chat' if is_native else '/v1/chat/completions'}"

    logger.info(f"Handling locally with {ROUTER_MODEL} (ctx: {ROUTER_CTX}, keep_alive: {keep_alive})")

    if not is_streaming:
        try:
            resp = await http_client.post(target_url, json=body, timeout=120.0)
            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code, content={"error": "Local inference failed."})
            data = resp.json()
            data["model"] = "Bob"
            if "id" in data:
                data["id"] = "chatcmpl-Bob"
            return JSONResponse(content=data)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Local inference failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    return StreamingResponse(
        _proxy_stream_local(target_url, body, is_native),
        media_type="application/x-ndjson" if is_native else "text/event-stream"
    )


def _is_expert_warm() -> bool:
    """Check if the desktop expert is still within its warm window."""
    return time.time() < _expert_warm_until


def _set_expert_warm(duration: int = 600):
    """Set the expert warm timer (default 10 minutes)."""
    global _expert_warm_until
    _expert_warm_until = time.time() + duration
    logger.info(f"Expert warm timer set for {duration}s.")


async def _forward_to_desktop(body: dict, path: str, is_native: bool, is_streaming: bool):
    """Forward a request to the desktop orchestrator, waking it via WoL if necessary."""

    # Ensure desktop is alive (wake + start if needed)
    if not await is_desktop_online():
        logger.info("Desktop offline for heavy request. Initiating wake sequence...")
        ready = await ensure_desktop_ready()
        if not ready:
            msg = (
                "⚠️ **Desktop is offline** and could not be woken up.\n\n"
                "Please check that:\n"
                "- Wake-on-LAN is enabled in BIOS\n"
                "- The desktop is connected via Ethernet\n"
                "- `WAKER_TOKEN` is correct in `.env`\n"
                "- SSH keys are set up (`ssh-copy-id`)"
            )
            return _command_response(msg, is_streaming, is_native)

    # Build target URL, preserving the original API path
    target_url = f"{DESKTOP_ORCHESTRATOR_URL}{path}"

    # Add routing header so the desktop skips its own triage
    headers = {
        "X-Forwarded-By-Router": "true",
        "Content-Type": "application/json"
    }

    logger.info(f"Forwarding to desktop: {target_url}")

    if not is_streaming:
        try:
            resp = await http_client.post(target_url, json=body, headers=headers, timeout=600.0)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.ConnectError:
            return JSONResponse(status_code=502, content={
                "error": "Desktop orchestrator is not responding."
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Desktop forward failed: {e}")
            return JSONResponse(status_code=502, content={"error": f"Desktop unreachable: {e}"})

    return StreamingResponse(
        _proxy_stream_desktop(target_url, body, is_native, headers),
        media_type="application/x-ndjson" if is_native else "text/event-stream"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Router Node on port {ROUTER_PORT}")
    logger.info(f"Local model: {ROUTER_MODEL} via {ROUTER_OLLAMA_URL}")
    logger.info(f"Desktop: {DESKTOP_ORCHESTRATOR_URL}")
    logger.info(f"Waker service: {WAKER_URL}")
    uvicorn.run(app, host="0.0.0.0", port=ROUTER_PORT, reload=False, log_level="warning")
