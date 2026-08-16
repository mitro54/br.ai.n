This document outlines the step-by-step process to build the Unified Local AI Workspace. 

---

## ⚡ Quick Start: Automatic Installer
If you are already in WSL2 with Docker and NVIDIA drivers installed, you can skip the manual steps and run the unified installer:

```bash
chmod +x setup_workspace.sh
./setup_workspace.sh
```

**For a Distributed (2-Device) Setup:**
If you are offloading the frontend router to a Raspberry Pi, run the Pi-specific installer on that device instead:
```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

---

**Prerequisite:** Please read `ARCHITECTURE.md` first to understand the VRAM constraints, network rules, and GPU Mutex locking requirements.

---

## Phase 1: Foundation (WSL2 & Docker)
1. **WSL2 Verification:** Ensure Ubuntu is installed and running on Windows 11. Open your Windows terminal and run:
```bash
   wsl --update
   wsl --set-default-version 2

```

2. **NVIDIA Drivers:** Verify the Windows NVIDIA driver is passing through to WSL2 by opening your Ubuntu terminal and running:
```bash
nvidia-smi

```


*You should see your NVIDIA GPU (e.g., RTX 3090, 4090) listed with its VRAM.*
3. **Docker Installation:** Install Docker Engine and the NVIDIA Container Toolkit inside WSL2 to allow containers to access the GPU.
*Note: Using Docker Desktop for Windows is acceptable, but native Docker inside WSL2 often yields better performance and resource control.*

---

## Phase 2: The LLM Engine (Ollama Configuration)

1. **Verify Ollama Installation:** Ensure the Ollama service is running in WSL2. 
2. **Pull and build models:** Ensure the resident router and optimized expert models are present.
   ```bash
   # Pull the resident router model
   ollama pull qwen2.5:1.5b

   # Pull the base 27B model
   ollama pull qwen3.8:27b

   # Create the optimized 65k context Expert model (recommended for 24GB VRAM)
   ollama create qwen3.8opt:latest -f Modelfile.qwen38opt
   ```

3. **Apply VRAM Guardrail:** You must set the environment variable globally in the Ollama systemd service to ensure models drop from memory.
3.1 Run: 
```bash
sudo systemctl edit ollama.service
```
3.2 Add this under the [Service] block:
```bash
Environment="OLLAMA_KEEP_ALIVE=0"
```
3.3 Then reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

---

## Phase 3: Orchestration & Search (Docker Compose)

1. **Create Network:** Set up the custom Docker bridge network.
```bash
docker network create ai-workspace-net
```

2. **SearXNG Config:** Create a folder named `searxng` and a file inside it named `settings.yml`. We will generate a secure key for it:
```bash
mkdir searxng
echo "use_default_settings: True
server:
  port: 8080
  bind_address: \"0.0.0.0\"
  secret_key: \"$(openssl rand -hex 32)\"
search:
  formats:
    - json" > searxng/settings.yml
```

3. **Deploy Core Stack:** Create a `docker-compose.yml` file with the following content:

```yaml
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    networks:
      - ai-workspace-net
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/

  open-webui:
    image: ghcr.io/open-webui/open-webui:latest
    container_name: open-webui
    networks:
      - ai-workspace-net
    ports:
      - "3000:8080"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - open-webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:8000
      - IMAGE_GENERATION_ENGINE=openai
      - OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
      - OPENAI_API_KEY=not-needed
    depends_on:
      - searxng

networks:
  ai-workspace-net:
    driver: bridge

volumes:
  open-webui_data:
```

4. **Launch Stack:**
```bash
docker compose up -d
```

---

## Phase 4: Image Generation Engine (ComfyUI)

1. **Environment:** Create a dedicated Python virtual environment natively in WSL2 to isolate dependencies and prevent pip conflicts.
```bash
python3 -m venv comfy-env
source comfy-env/bin/activate

```


2. **Installation:** Clone ComfyUI and install requirements, ensuring PyTorch with CUDA support is installed.
```bash
git clone [https://github.com/comfyanonymous/ComfyUI.git](https://github.com/comfyanonymous/ComfyUI.git)
cd ComfyUI
pip install torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install -r requirements.txt

```


3. **Configuration & Launch:** Launch ComfyUI with appropriate memory settings for your GPU. For a 24GB+ card (RTX 3090/4090/5090), `--normalvram` is recommended:
```bash
python main.py --normalvram --listen 0.0.0.0
```

4. **Export API Workflow:** Open the ComfyUI web interface (`http://localhost:8188`), build your favorite workflow, and click **"Save (API Format)"**. Save this as `workflow_api.json` in your project root.
   * *Note: You must enable "Enable Dev mode" in ComfyUI settings to see the API export button.*

5. **Integration (The Orchestrator Proxy):** This provides the Mutex lock and the "Fast Orchestrator" tier. Create `orchestrator.py`:

```python
import asyncio
import httpx
import os
import json
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")

app = FastAPI(title="AI Workspace Orchestrator Proxy")
gpu_lock = asyncio.Lock()

# Model Configs
ROUTER_MODEL = "qwen2.5:1.5b"
EXPERT_MODEL = "qwen3.8:27b"
OLLAMA_URL = "http://localhost:11434"
COMFYUI_URL = "http://localhost:8188"
WORKFLOW_PATH = "workflow_api.json"

async def get_intent(prompt: str) -> str:
    """Uses the fast router model to categorize the intent."""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": ROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "Classify intent: FAST (chat), EXPERT (coding/complex), or IMAGE. Reply with a single word."}, 
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "keep_alive": -1  # Keep router pinned to VRAM
        }
        try:
            resp = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload, timeout=2.0)
            resp.raise_for_status()
            intent = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "EXPERT").upper()
            logger.info(f"Intent classified as: {intent}")
            return intent
        except Exception as e:
            logger.warning(f"Intent routing failed: {e}. Defaulting to EXPERT.")
            return "EXPERT"

async def stream_proxy(url: str, body: dict, req_lock: asyncio.Lock = None):
    """Proxies the request, conditionally locking the GPU."""
    async with httpx.AsyncClient(timeout=None) as client:
        if req_lock:
            async with req_lock:
                logger.info("Expert GPU Lock Acquired.")
                async with client.stream("POST", url, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                logger.info("Expert GPU Lock Released.")
        else:
            logger.info("Fast Path execution (No lock).")
            async with client.stream("POST", url, json=body) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

@app.post("/v1/chat/completions")
async def proxy_ollama(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])
        user_prompt = messages[-1].get("content", "") if messages else ""
        
        # 1. Check for Multimedia (Vision) logic
        has_images = any("image_url" in str(msg) for msg in messages)
        intent = await get_intent(user_prompt)
        
        # 2. Expert Path Escalation: Vision, Large Context or Orchestrator decision
        is_expert = has_images or "EXPERT" in intent or len(user_prompt) > 2000
        target_model = EXPERT_MODEL if is_expert else ROUTER_MODEL
        body["model"] = target_model
        
        logger.info(f"Routing request to model: {target_model}")
        
        return StreamingResponse(
            stream_proxy(f"{OLLAMA_URL}/v1/chat/completions", body, req_lock=gpu_lock if is_expert else None),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Orchestrator Chat Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Orchestrator Chat Error: {str(e)}"})

@app.post("/v1/audio/transcriptions")
async def proxy_transcription(request: Request):
    """Pass through to Ollama's Whisper endpoint without locking the GPU."""
    try:
        async with httpx.AsyncClient() as client:
            form_data = await request.form()
            file = form_data.get("file")
            if not file:
                return JSONResponse(status_code=400, content={"error": "No file provided"})
            
            files = {"file": (file.filename, file.file, file.content_type)}
            resp = await client.post(f"{OLLAMA_URL}/v1/audio/transcriptions", files=files, data=form_data)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        logger.error(f"Transcription Proxy Error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Transcription Error: {str(e)}"})

@app.post("/v1/images/generations")
async def proxy_comfyui(request: Request):
    async with gpu_lock:
        try:
            logger.info("ComfyUI GPU Lock Acquired. Generating Image.")
            body = await request.json()
            prompt_text = body.get("prompt", "")
            
            with open(WORKFLOW_PATH, 'r') as f:
                workflow = json.load(f)
            
            # Inject prompt into the node that handles text encoding
            for node_id in workflow:
                if workflow[node_id].get("class_type") == "CLIPTextEncode":
                    workflow[node_id]["inputs"]["text"] = prompt_text
                    break

            async with httpx.AsyncClient() as client:
                p_resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=5.0)
                p_resp.raise_for_status()
                p_id = p_resp.json().get("prompt_id")
                
                # Poll history with a retry limit
                retries = 0
                max_retries = 60  # Wait up to 60 seconds
                while retries < max_retries:
                    h_resp = await client.get(f"{COMFYUI_URL}/history/{p_id}")
                    history = h_resp.json()
                    
                    if p_id in history:
                        outputs = history[p_id].get("outputs", {})
                        if outputs:
                            node_id = list(outputs.keys())[0]
                            images = outputs[node_id].get("images", [])
                            if images:
                                filename = images[0].get("filename")
                                logger.info(f"Image generated: {filename}")
                                # In a real implementation, you'd return the actual image bytes or URL.
                                # Assuming Open WebUI requires standard OpenAI schema output:
                                return JSONResponse(content={
                                    "created": 1,
                                    "data": [{"url": f"{COMFYUI_URL}/view?filename={filename}"}]
                                })
                        break
                    
                    await asyncio.sleep(1)
                    retries += 1
                
                if retries >= max_retries:
                    raise TimeoutError("ComfyUI generation timed out.")
                    
        except Exception as e:
            logger.error(f"ComfyUI Proxy Error: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"ComfyUI Proxy Error: {str(e)}"})
        finally:
            logger.info("ComfyUI GPU Lock Released.")

if __name__ == "__main__":
    import uvicorn
    # Standard parameters for ASGI production stability
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=8000, reload=False, log_level="info", access_log=False)
```

6. **Run Orchestrator:**
```bash
python orchestrator.py
```

---

## Phase 5: Local Network Accessibility (Multi-Device Access)

To access the workspace from your phone, tablet, or another device on your WiFi, you must bridge the WSL2 environment to your local network.

### 1. Enable WSL2 Mirrored Networking
This is the most reliable way to expose WSL2 services to your LAN without complex port forwarding.
1.1 On your **Windows Host**, create or edit `%USERPROFILE%\.wslconfig`.
1.2 Add the following configuration:
```ini
[wsl2]
networkingMode=mirrored
firewall=true
```
1.3 Restart WSL: Open PowerShell and run `wsl --shutdown`, then reopen your terminal.

### 2. Open Windows Firewall
You must allow inbound traffic on the WebUI port (3000). Run this in **PowerShell as Administrator**:
```powershell
New-NetFirewallRule -DisplayName "AI Workspace - Open WebUI" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow
```

### 3. Find Your Desktop's Local IP
In your Windows terminal (CMD or PowerShell), run:
```cmd
ipconfig
```
Look for the **IPv4 Address** under your active network adapter (e.g., `192.168.1.50`).

### 4. Connect from Other Devices
On your phone or tablet:
4.1 Ensure you are on the same WiFi as the desktop.
4.2 Open your mobile browser and navigate to: `http://<YOUR_IP>:3000`.

---

## Phase 6: VRAM Stress Testing & Validation

Run these tests sequentially while monitoring `nvtop` or `nvidia-smi` in a separate terminal.

1. **Test 1 (Baseline):** Boot the system. Verify idle VRAM is ~2.8GB.
2. **Test 2 (RAG & Web Search):** Trigger a complex web search via Open WebUI.
* *Expected Result:* VRAM spikes to ~18GB (Qwen + Embedding model) and drops back down immediately after the text output finishes.


3. **Test 3 (Image Generation):** Request an image via Open WebUI prompt.
* *Expected Result:* ComfyUI loads into the empty VRAM (~12GB used), generates the image, and clears out immediately.

4. **Test 4 (The Mutex Check):** Attempt to send a text prompt *while* an image is generating.
* *Expected Result:* The system should queue the text request rather than crashing the RTX 4090 with an OOM error. The GPU lock ensures serialized access to the Expert/Media slice.

---

## Phase 7: Automating Development & CI/CD Pipeline

To ensure the `orchestrator.py` script and Docker deployment remain stable as you build, a CI/CD pipeline should be implemented.

**If using GitHub for version control:**
1. Create `.github/workflows/ci.yml`.
2. **Linting:** Use `ruff` or `flake8` to validate `orchestrator.py` for Python syntax errors.
3. **Type Checking:** Use `mypy` to ensure async logic is properly typed.
4. **Docker Confidence:** Add a step to run `docker compose build` to verify the syntax and dependencies of the Core Stack.

**Example `ci.yml` Fragment:**
```yaml
name: CI
on: [push, pull_request]

jobs:
  test-proxy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install ruff fastapi uvicorn httpx
    - name: Lint with Ruff
      run: ruff check .
```
