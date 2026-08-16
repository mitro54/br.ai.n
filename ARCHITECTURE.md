# 🧠 Project Architecture: Agentic Local AI Orchestrator (Bob)

## 1. System Context & Goals
**Goal:** Build a fully localized, unified, and agent-driven AI workspace ("Bob") running on a single NVIDIA GPU (24GB+ VRAM recommended) via Windows Subsystem for Linux (WSL2).
**Scalability:** While optimized for 24GB VRAM (allowing for large expert models like Qwen 27B), the system can be scaled down to 8GB-12GB VRAM by selecting smaller expert models (e.g., Llama-3-8B).
**Capabilities:** Text generation, complex coding, vision (image analysis), voice transcription, document RAG, live web search, and image generation.
**Accessibility:** The workspace must be accessible from the local area network (LAN), allowing phones, tablets, and other laptops to utilize the host's GPU resources via a unified web interface.
**Key Constraint:** The system must use a **Tiered Orchestration** approach to provide instant responses for simple tasks while dynamically routing heavy tasks to expert models, managing VRAM aggressively to prevent OOM on a 24GB budget.

### Distributed Two-Node Architecture
The system supports a split deployment across two physical nodes:

* **Router Node (Raspberry Pi 5, 8GB):** Always-on network entry point. Runs `router.py`, Open WebUI, SearXNG, and the local triage model (`qwen2.5:1.5b` via Ollama on CPU). Handles simple queries locally and forwards complex tasks to the Desktop.
* **Desktop Node (RTX 4090, 24GB VRAM):** GPU powerhouse. Runs `orchestrator.py`, the Expert model (35B), ComfyUI, and the cline-builder pipeline. Can be asleep — the Pi wakes it via Wake-on-LAN when heavy tasks arrive.
* **Communication:** The Pi adds an `X-Forwarded-By-Router: true` header when proxying requests to the Desktop. The Desktop detects this header and skips its own triage, routing directly to the Expert model.
* **Standalone Mode:** The Desktop's `orchestrator.py` works independently without the Pi. The header check is simply ignored when not present.

---

## 2. Hardware Guardrails & VRAM Budget
The system operates on a strict hardware limit. All agentic routing and tool execution must respect the following constraints:

* **GPU:** 1x NVIDIA RTX 4090 (24GB VRAM). 
* **OS:** Windows 11 Host running WSL2 (Ubuntu).
* **VRAM Baseline Allocation:**
  * Windows OS / Display / System: ~2.5GB 
  * Resident Orchestrator (Ollama - `qwen2.5:1.5b`): ~1.2GB
  * **Available Expert VRAM:** ~20GB
* **Expert Model Budget (RTX 4090 Optimized):**
  * **Model Weights (Q3_K_XL 35B-A3B):** ~15.7GB
  * **KV Cache (256k Context @ Q8_0):** ~5.4GB (utilizing GQA and KV Quantization)
  * **Total:** ~21.1GB (Fits comfortably in 24GB VRAM)
* **Rule:** The Resident Orchestrator stays pinned. The Expert LLM and Visual/Image models **cannot** exist in VRAM simultaneously and must be hard-swapped via the GPU Mutex.

---

## 3. Architecture Decision Records (ADRs)

### ADR 001: Time-Slicing VRAM over Static Loading
* **Context:** We lack the VRAM to keep both the primary reasoning LLM and the image generation model in memory.
* **Decision:** We will aggressively time-slice VRAM. Ollama will be configured with `OLLAMA_KEEP_ALIVE=0` (or `1m`), forcing immediate model unloading after inference. ComfyUI will run with `--lowvram` to offload weights immediately to system RAM after generating an image.

### ADR 002: GPU Mutex via Intelligent Proxy (The "Orchestrator")
* **Context:** Independent services (Ollama, ComfyUI) cannot coordinate VRAM usage natively, and reloading large models causes lag.
* **Decision:** We will implement a **Python FastAPI Proxy** (The Orchestrator) using a tiny resident LLM as a router.
*   **Context:** Independent services (Ollama, ComfyUI) cannot coordinate VRAM usage natively, and reloading large models causes lag.
*   **Decision:** We will implement a **Python FastAPI Proxy** (The Orchestrator) using a tiny resident LLM as a router.
    *   All API calls pass through this proxy.
    *   The proxy uses the **Resident Orchestrator** to categorize requests (Fast Path vs. Expert Path).
    *   The proxy maintains an `asyncio.Lock` for the GPU's "Expert Zone" (the remaining ~20GB of VRAM).
    *   Fast Path requests (greetings, simple knowledge) are answered by the Orchestrator immediately.
    *   Expert Path requests (Coding, Vision, RAG) or Image Generations trigger the Mutex lock and model swap.

### ADR 005: Managed Subprocess Lifecycle for llama.cpp
*   **Context:** Native Ollama is great for general tasks, but `llama-server` provides superior control over HuggingFace models, GGUF quants, and KV cache settings.
*   **Decision:** The Orchestrator will act as a **Process Manager** for `llama-server`. 
    *   It spawns instances on-demand with specific flags (`-hf`, `-c`, `-np`).
    *   It uses `asyncio.Event` readiness signals and background health-polling to ensure the server is fully up before routing traffic.
    *   It provides internal APIs (`/internal/model/load`) to allow Dockerized pipelines to trigger host-level model swaps.

### ADR 006: Extended Memory via KV Quantization
*   **Context:** Standard 16-bit KV caches consume ~10GB+ at 128k context, leading to OOM on 24GB cards when combined with 35B models.
*   **Decision:** We enforce **Q8_0 KV Cache Quantization** and **Flash Attention** across all expert roles. This halves the memory footprint of the context window, allowing for 256k tokens to fit in the same space previously required for 128k.

### ADR 004: ComfyUI-to-OpenAI Bridge (Prompt-to-Graph)
*   **Context:** Open WebUI expects DALL-E (OpenAI) schema, while ComfyUI requires a workflow graph JSON.
*   **Decision:** The Orchestrator Proxy will perform **Prompt-to-Graph injection**. 
    *   A template `workflow_api.json` (exported from ComfyUI in API mode) will be stored on disk.
    *   The proxy will read this JSON, inject the user's prompt into the correct node (usually `CLIPTextEncode`), post it to ComfyUI's `/prompt` endpoint, and poll the `/history` endpoint for the resulting image URL.

---

## 4. Core Component Definitions

### A. LLM Engines: The Universal Trio
* **Ollama (Native):** Daily driver for resident routing and general chat.
* **llama.cpp (Managed):** Power-user engine for HuggingFace models and extreme context coding (128k+).
* **LM Studio (External):** Integration via `lms` CLI for easy model discovery and experimentation.
* **Expert Model:** `unsloth/Qwen3.6-35B-A3B` (MoE architecture optimized for coding and long context).

### B. The Orchestrator & Frontend: Open WebUI
* **Role:** Unified UI, RAG document processing, voice transcription (Whisper), embedding generation, and agentic tool routing.
* **Deployment:** Docker container on `ai-workspace-net`.
* **Agentic Routing:** Utilizes Open WebUI's "Tools" feature to detect intent and route requests to SearXNG or ComfyUI.

### C. Live Web Search & RAG: SearXNG + Open WebUI
* **Role:** 
  * **SearXNG:** Live web search for real-time grounding.
  * **Open WebUI:** Handles document RAG, PDF parsing, and image/video preprocessing before sending context to the Orchestrator. 
* **Deployment:** Docker containers on `ai-workspace-net`.

### D. Image Generation Engine: ComfyUI
* **Role:** Node-based backend for text-to-image workflows.
* **Deployment:** Python `venv` natively in WSL2.
*   **Role:** Node-based backend for text-to-image workflows.
*   **Deployment:** Python `venv` natively in WSL2.
*   **Configuration:**
    *   **Configuration & Launch:** Launch ComfyUI with appropriate memory settings for your GPU. For a 24GB+ card (RTX 3090/4090/5090), `--normalvram` is recommended:
        ```bash
        python main.py --normalvram --listen 0.0.0.0
        ```
    *   Integration: Accessed via the **Orchestrator Proxy** to ensure VRAM safety.

---

## 5. Data Flow & VRAM Orchestration Logic
This flow illustrates the VRAM safety logic when the LLM triggers a heavy tool (Image Generation).

4. **Model Loading:** The Expert model or ComfyUI loads into the remaining ~20GB VRAM.
5. **Selective Interception:** Automatically silences automated image descriptions and expansion pings to prevent VRAM signal "leakage" to the Expert model.
6. **Auto-Unload:** Once finished, the expert model drops (`KEEP_ALIVE=0`). 
7. **Release:** Orchestrator releases the lock. The system returns to the Resident Orchestrator baseline.
8. **Periodic Sweep**: A background task sweeps idle ComfyUI RAM/VRAM every 5 minutes.
9. **Project Extraction (`!move`)**: The `mover.py` module parses conversation history to reconstruct file systems. It uses regex-based extraction to separate code from commands and sanitizes paths (clears `..`) to ensure they stay within the guest-workspace boundary.
10. **Dynamic Settings Fallback**: The Orchestrator applies specialized `temperature` and `presence_penalty` values only to the default `qwen3.8:27b` (or `qwen3.8opt:latest`) model. For any third-party expert model, it reverts to standard defaults to ensure stability.

---

## 6. Network & Port Mapping Matrix

| Component | Internal Network DNS | Host Exposed Port | Protocol |
| :--- | :--- | :--- | :--- |
| **Open WebUI** | `open-webui:8080` | `3000` | HTTP |
| **Ollama** | `ollama:11434` | `None` (Internal only) | REST/HTTP |
| **SearXNG** | `searxng:8080` | `None` (Internal only) | JSON/HTTP |
| **ComfyUI** | `127.0.0.1:8188` (Host routed) | `8188` | REST/WS |

*Note: ComfyUI is run outside the core Docker network for easier access to local model `.safetensors` directories, and is accessed via the host IP.*
---

## 7. Autonomous Build Pipeline (The Distiller)

The system includes a dedicated **Dockerized Build Pipeline** (`cline-builder`) for fully autonomous project implementation.

### A. The 4-Pass Distillation Engine
To prevent context window saturation and ensure high-quality code, the pipeline uses a tiered distillation process:
1. **Architect Pass:** Analyzes the conversation to define business goals and directory structures.
2. **Engineer Pass:** Maps logic to files and defines design patterns and build orders.
3. **Test Engineer Pass:** Identifies edge cases and defines verification gates.
4. **Safety Inspector Pass:** Audits the plan for security vulnerabilities and resource safety.

### B. Execution Flow
- **Input:** A `.build_conversation.json` file exported by the `!build` command.
- **Processing:** The `distill.py` engine runs the 4 passes sequentially, loading/unloading models for VRAM efficiency.
- **Output:** A structured `.clinerules` file embedded with critical directives and implementation roadmaps.
- **Implementation:** A `Cline` agent (using the `unsloth/Qwen3.6-35B-A3B:Q3_K_XL` model via `llama-server`) reads the rules and executes code autonomously within a **256k context window**.

### C. Build Constraints
- **Anti-Loop Shield:** The builder is forbidden from attempting the same bug fix more than twice.
- **Chunked Ingestion:** Long conversations are automatically split into 2k-token "extraction chunks" to prevent CPU ingestion stalls during the distillation process.
