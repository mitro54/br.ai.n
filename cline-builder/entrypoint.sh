#!/bin/bash
set -euo pipefail

# =============================================================================
# Multi-Agent Cline Builder Pipeline — Entrypoint
# =============================================================================
# Three-phase pipeline:
#   Phase 1: Multi-pass context distillation (Architect → Engineer → Safety)
#   Phase 2: Iterative Cline CLI build/verify/safety cycle
#
# Environment Variables (set by orchestrator):
#   CONVERSATION_FILE  - Path to conversation JSON
#   AGENT_CONFIG_PATH  - Path to agent_config.json
#   OLLAMA_HOST        - Ollama API URL
#   EXPERT_CTX         - Context window size for distillation
#   CLINE_CTX          - Context window size for Cline agent
#   CLINERULES_PATH    - Output path for .clinerules
# =============================================================================

# Set the global config directory so ALL cline commands use it
export CLINE_DIR="/root/.config/Cline"
export NODE_NO_WARNINGS=1

# --- VRAM Safety Guard ---
# Ensure that we release the GPU expert model when the container shuts down
# regardless of success or failure.
cleanup_vram() {
    echo ""
    echo "========================================"
    echo "🧹 VRAM VACUUM: Releasing GPU Expert..."
    echo "========================================"
    # Send shutdown signal to orchestrator
    ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://host.docker.internal:8000}"
    curl -s -X POST "${ORCHESTRATOR_URL}/v1/shutdown_expert" > /dev/null || true
    echo "  ✓ Expert unloaded."
}
trap cleanup_vram EXIT INT TERM

CONFIG_PATH="${AGENT_CONFIG_PATH:-/app/agent_config.json}"
CONVERSATION_FILE="${CONVERSATION_FILE:-/workspace/.cline_context/conversation.json}"
CLINERULES_PATH="${CLINERULES_PATH:-/workspace/.clinerules}"
OLLAMA_HOST="${OLLAMA_HOST:-http://host.docker.internal:11434}"
CLINE_CTX="${CLINE_CTX:-65536}"
EXPERT_CTX="${EXPERT_CTX:-65536}"
CLINE_BIN="$(which cline 2>/dev/null || echo /usr/local/bin/cline)"

echo "========================================"
echo "🔨 Cline Builder Pipeline"
echo "========================================"
echo "  Config:       ${CONFIG_PATH}"
echo "  Conversation: ${CONVERSATION_FILE}"
echo "  Ollama:       ${OLLAMA_HOST}"
echo "  Distill CTX:  ${EXPERT_CTX}"
echo "  Cline CTX:    ${CLINE_CTX}"
echo "  Project:      ${PROJECT_NAME:-<unnamed>}"
echo "  Timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================"

# Setup Directories
mkdir -p /workspace/.cline_context
mkdir -p /workspace/.cline_logs

# Migrate legacy structures from older pipelines (backwards compatibility)
mv /workspace/.build_log_iter_*.txt /workspace/.cline_logs/ 2>/dev/null || true
mv /workspace/.verify_log_iter_*.txt /workspace/.cline_logs/ 2>/dev/null || true
mv /workspace/.safety_log_iter_*.txt /workspace/.cline_logs/ 2>/dev/null || true
mv /workspace/.distill_*.md /workspace/.cline_context/ 2>/dev/null || true
if [ -f "/workspace/.build_conversation.json" ]; then
    # Move the old conversation out of the root, but don't overwrite the new one 
    mv -n /workspace/.build_conversation.json /workspace/.cline_context/legacy_conversation.json 2>/dev/null || true
fi
if [ -f "/workspace/.build_issues.md" ] && [ ! -f "/workspace/.cline_context/.build_issues.md" ]; then
    mv /workspace/.build_issues.md /workspace/.cline_context/ 2>/dev/null || true
fi

# Clear previous run artifacts to ensure no confusion
echo "🧹 Cleaning previous run logs and distillation files..."
rm -f /workspace/.cline_logs/*.txt
rm -f /workspace/.cline_context/distill_*.md
rm -f /workspace/.build_complete

# --- Noise Suppression Bootstrap ---
# Ensure node_modules and metadata are physically ignored by the agent's tools
echo "🚩 Bootstrapping noise suppression (.gitignore)..."
{
    echo "node_modules/"
    echo ".git/"
    echo ".venv/"
    echo "venv/"
    echo ".cline_logs/"
    echo ".cline_context/"
    echo ".knowledge_base/"
    echo "__pycache__/"
    echo ".pytest_cache/"
    echo "*.log"
} >> /workspace/.gitignore_builder

# Sort and unique the gitignore if it exists, otherwise use our builder version
if [ -f "/workspace/.gitignore" ]; then
    sort -u /workspace/.gitignore /workspace/.gitignore_builder -o /workspace/.gitignore
else
    cp /workspace/.gitignore_builder /workspace/.gitignore
fi
rm /workspace/.gitignore_builder

# --- Prerequisite Checks ---

if [ ! -f "$CONVERSATION_FILE" ]; then
    echo "✗ FATAL: Conversation file not found: ${CONVERSATION_FILE}"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "✗ FATAL: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

# Wait for backend providers to be reachable (up to 30 seconds each)
echo ""
echo "🔌 Checking backend connectivity..."

# Extract all unique base_urls from the config (handles both string and object model formats)
# For string models, the default is ollama_host; for objects, use base_url
DEFAULT_HOST=$(jq -r '.ollama_host // "http://host.docker.internal:11434"' "$CONFIG_PATH")
ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://host.docker.internal:8000}"

# Get base_urls paired with their provider type (skip llamacpp — orchestrator manages those on demand)
PROVIDER_URLS=$(jq -r --arg dh "$DEFAULT_HOST" '
  [.models | to_entries[] | .value |
    if type == "object" then
      select(.provider != "llamacpp") | .base_url // $dh
    else $dh
    end
  ] | unique | .[]' "$CONFIG_PATH" 2>/dev/null || echo "$DEFAULT_HOST")

# Check if any models use llamacpp (need orchestrator connectivity instead)
HAS_LLAMACPP=$(jq -r '[.models | to_entries[] | .value | select(type == "object" and .provider == "llamacpp")] | length' "$CONFIG_PATH" 2>/dev/null || echo "0")

for BASE_URL in $PROVIDER_URLS; do
    # Determine the health endpoint based on the URL
    # Ollama uses /api/tags, OpenAI-compatible uses /v1/models
    HEALTH_URL="${BASE_URL}/api/tags"
    if [[ "$BASE_URL" != *":11434"* ]]; then
        HEALTH_URL="${BASE_URL}/v1/models"
    fi

    echo "  Checking ${BASE_URL}..."
    RETRIES=0
    MAX_RETRIES=15
    until curl -sf "${HEALTH_URL}" > /dev/null 2>&1; do
        RETRIES=$((RETRIES + 1))
        if [ $RETRIES -ge $MAX_RETRIES ]; then
            echo "  ⚠ WARNING: Cannot reach ${BASE_URL} after ${MAX_RETRIES} attempts (continuing anyway)"
            break
        fi
        echo "    Waiting... (${RETRIES}/${MAX_RETRIES})"
        sleep 2
    done
    if [ $RETRIES -lt $MAX_RETRIES ]; then
        echo "  ✓ ${BASE_URL} is reachable"
    fi
done

# If any models use llamacpp, verify the orchestrator is reachable (it manages llama-server lifecycle)
if [ "$HAS_LLAMACPP" -gt 0 ]; then
    echo "  Checking orchestrator (manages llamacpp)..."
    RETRIES=0
    MAX_RETRIES=10
    until curl -sf "${ORCHESTRATOR_URL}/health" > /dev/null 2>&1; do
        RETRIES=$((RETRIES + 1))
        if [ $RETRIES -ge $MAX_RETRIES ]; then
            echo "  ⚠ WARNING: Cannot reach orchestrator at ${ORCHESTRATOR_URL} (llamacpp models may fail)"
            break
        fi
        echo "    Waiting for orchestrator... (${RETRIES}/${MAX_RETRIES})"
        sleep 2
    done
    if [ $RETRIES -lt $MAX_RETRIES ]; then
        echo "  ✓ Orchestrator reachable (will manage llamacpp on demand)"
    fi
fi

# --- Read Config ---
MAX_SIZE_MB=$(jq -r '.limits.max_project_size_mb // 2048' "$CONFIG_PATH")
MAX_ITERATIONS=$(jq -r '.limits.max_build_iterations // 5' "$CONFIG_PATH")
CLINE_MAX_TURNS=$(jq -r '.limits.cline_max_turns // 100' "$CONFIG_PATH")
# Extract cline model: handle both string ("model_name") and object ({"model": "..."}) formats
CLINE_MODEL=$(jq -r 'if (.models.cline | type) == "object" then .models.cline.model else (.models.cline // "qwen3.5:27b") end' "$CONFIG_PATH")
CLINE_PROVIDER=$(jq -r 'if (.models.cline | type) == "object" then (.models.cline.provider // "ollama") else "ollama" end' "$CONFIG_PATH")
CLINE_BASE_URL=$(jq -r --arg dh "$DEFAULT_HOST" 'if (.models.cline | type) == "object" then (.models.cline.base_url // $dh) else $dh end' "$CONFIG_PATH")
CLINE_STARTUP=$(jq -r '.cline_startup_message // "Read .clinerules and execute all tasks."' "$CONFIG_PATH")

# --- Project Size Check ---
check_project_size() {
    local dir_size_mb
    dir_size_mb=$(du -sm /workspace 2>/dev/null | cut -f1)
    echo "  📦 Workspace size: ${dir_size_mb} MB / ${MAX_SIZE_MB} MB limit"
    if [ "$dir_size_mb" -gt "$MAX_SIZE_MB" ]; then
        echo "✗ FATAL: Workspace exceeds size limit (${dir_size_mb} MB > ${MAX_SIZE_MB} MB)"
        return 1
    fi
    return 0
}

# =============================================================================
# PHASE 1: Multi-Pass Context Distillation
# =============================================================================
echo ""
echo "========================================="
echo "📚 Phase 1: Context Distillation (4-pass)"
echo "   Workspace: $(pwd)"
echo "========================================="

# Run distillation with unbuffered output
PYTHONUNBUFFERED=1 python3 /app/distill.py
DISTILL_EXIT=$?

if [ $DISTILL_EXIT -ne 0 ]; then
    echo "✗ FATAL: Distillation failed (exit code ${DISTILL_EXIT})"
    exit 1
fi

if [ ! -f "$CLINERULES_PATH" ]; then
    echo "✗ FATAL: .clinerules file was not created"
    exit 1
fi

echo ""
echo "  ✓ .clinerules generated ($(wc -c < "$CLINERULES_PATH") bytes)"

# Reset completion state in case this is a rebuild
rm -f /workspace/.build_complete

# =============================================================================
# GIT SAFETY NET (Component 7)
# =============================================================================
setup_git_safety() {
    cd /workspace
    
    # Try to initialize if missing, but don't fail if we can't
    if [ ! -d ".git" ]; then
        # Project has no git — initialize for local snapshot only
        if git init -q 2>/dev/null; then
            git config user.email "builder@local"
            git config user.name "Cline Builder"
            git add -A 2>/dev/null
            git commit -q -m "snapshot: pre-build state" 2>/dev/null
            echo "  📸 Created local snapshot (no remote, no pushing)"
        else
            echo "  ⚠️ Skipping git safety net (not a repository and cannot initialize)"
            return 0
        fi
    fi
    
    # Final check to ensure we are in a working tree
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 0
    fi
    
    # Always work on a branch, never on main/master
    local MAIN_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
    local BRANCH_NAME="agent/build-$(date +%s)"
    git checkout -b "$BRANCH_NAME" 2>/dev/null || true
    echo "  🌿 Working on branch: ${BRANCH_NAME} (based on ${MAIN_BRANCH})"
    echo "  💡 To review: git diff ${MAIN_BRANCH}"
    echo "  💡 To rollback: git checkout ${MAIN_BRANCH}"
}

echo ""
echo "🌿 Setting up git safety net..."
setup_git_safety

# =============================================================================
# SESSION STATE GENERATOR (Component 3)
# =============================================================================
generate_session_state() {
    local ITERATION=$1
    local STEP=$2
    local STATE_FILE="/workspace/.cline_context/.session_state.md"
    
    echo "# Session State (Auto-generated)" > "$STATE_FILE"
    echo "" >> "$STATE_FILE"
    echo "## Current Position" >> "$STATE_FILE"
    echo "- **Iteration**: ${ITERATION}/${MAX_ITERATIONS}" >> "$STATE_FILE"
    echo "- **Step**: ${STEP}" >> "$STATE_FILE"
    echo "- **Timestamp**: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE_FILE"
    echo "" >> "$STATE_FILE"
    
    # Inject known issues if they exist
    if [ -f "/workspace/.cline_context/.build_issues.md" ]; then
        echo "## Known Issues (from previous steps)" >> "$STATE_FILE"
        cat /workspace/.cline_context/.build_issues.md >> "$STATE_FILE"
        echo "" >> "$STATE_FILE"
    fi
    
    # Inject quality audit if it exists
    if [ -f "/workspace/.cline_context/quality_audit.md" ]; then
        echo "## 🛡️ Architectural & Quality Critique" >> "$STATE_FILE"
        echo "> These notes represent the project's quality conscience. Address these critiques before implementation." >> "$STATE_FILE"
        cat /workspace/.cline_context/quality_audit.md >> "$STATE_FILE"
        echo "" >> "$STATE_FILE"
    fi

    # Inject analysis notes if agent wrote any
    if [ -f "/workspace/.cline_context/analysis_notes.md" ]; then
        echo "## Agent Discovery Notes" >> "$STATE_FILE"
        tail -c 3000 /workspace/.cline_context/analysis_notes.md >> "$STATE_FILE"
        echo "" >> "$STATE_FILE"
    fi
    
    # Inject summaries from previous step logs
    echo "## Previous Step Summaries" >> "$STATE_FILE"
    for log in /workspace/.cline_logs/*.txt; do
        if [ -f "$log" ]; then
            local LOG_NAME=$(basename "$log")
            echo "### ${LOG_NAME}" >> "$STATE_FILE"
            grep -iE "(FINAL SUMMARY|attempt_completion|✓|✗|ERROR|TODO|BLOCKED)" "$log" 2>/dev/null \
                | tail -10 >> "$STATE_FILE" || true
            echo "" >> "$STATE_FILE"
        fi
    done
}

# =============================================================================
# PHASE 2: Iterative Cline Build Cycle
# =============================================================================
echo ""
echo "========================================="
echo "🤖 Phase 2: Cline Build Cycle"
echo "   Output: /workspace"
echo "========================================="
echo "  Model:          ${CLINE_MODEL}"
echo "  Max iterations: ${MAX_ITERATIONS}"
echo "  Max turns/iter: ${CLINE_MAX_TURNS}"
echo ""

# --- Phase 2 Setup: Auto-Auth for CLI ---
# Determine the API base URL for Cline (always use /v1 for OpenAI-compatible auth)
if [ "$CLINE_PROVIDER" = "ollama" ]; then
    CLINE_AUTH_URL="${CLINE_BASE_URL}/v1"
else
    CLINE_AUTH_URL="${CLINE_BASE_URL}/v1"
fi
echo "  🔑 Configuring provider (${CLINE_PROVIDER}) for Cline CLI..."

"$CLINE_BIN" auth \
    -p openai-compatible \
    -k "dummy" \
    -m "$CLINE_MODEL" \
    -b "${CLINE_AUTH_URL}"

ITERATION=0
BUILD_COMPLETE=false

while [ $ITERATION -lt $MAX_ITERATIONS ] && [ "$BUILD_COMPLETE" = false ]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "─── Iteration ${ITERATION}/${MAX_ITERATIONS} ───"

    # Pre-flight size check
    if ! check_project_size; then
        echo "✗ Build aborted: project size limit exceeded"
        exit 1
    fi

# --- Build Phase ---
    echo "  🔧 Running Cline (Build mode)..."
    generate_session_state "$ITERATION" "build"

    CURRENT_TIMEOUT=1800 # 30 minutes
    BUILD_MSG="IMPORTANT: You are running in a headless autonomous CI runner. NEVER call ask_question or ask_followup_question. First read '.cline_context/.session_state.md' to understand what has been done so far. Then read '.clinerules' and execute remaining implementation tasks. Make all technical decisions independently."

    if [ $ITERATION -eq $MAX_ITERATIONS ]; then
        echo "  🚨 FINAL ROUND: Shifting to Stabilization and Debugging..."
        BUILD_MSG="IMPORTANT: Headless runner (DO NOT ask questions). First read '.cline_context/.session_state.md'. CRITICAL: This is the FINAL iteration (${ITERATION} of ${MAX_ITERATIONS}). Your directive is now STABILIZATION. Revisit any TODOs, uncommented code, or failing tests. Fix the root causes of any remaining bugs."
        CURRENT_TIMEOUT=1800
    elif [ $ITERATION -gt 1 ]; then
        BUILD_MSG="IMPORTANT: Headless runner (DO NOT ask questions). First read '.cline_context/.session_state.md' to recover your memory. Continue building the project. Review what was done in the previous iteration, fix any issues, and complete remaining tasks from .clinerules. This is iteration ${ITERATION} of ${MAX_ITERATIONS}. Remember: keep momentum and don't get stuck on one bug."
    fi

    set +e
    export CLINE_BIN CLINE_MODEL CURRENT_TIMEOUT BUILD_MSG
    script -q -e -c '"$CLINE_BIN" -v --auto-approve true \
        -P openai-compatible \
        -m "$CLINE_MODEL" \
        --timeout "$CURRENT_TIMEOUT" \
        "$BUILD_MSG"' \
        "/workspace/.cline_logs/build_log_iter_${ITERATION}.txt"
    CLINE_EXIT=$?
    set -e

    echo "  ↳ Cline exited with code ${CLINE_EXIT}"

    # Post-build size check
    if ! check_project_size; then
        echo "✗ Build aborted: project grew beyond size limit"
        exit 1
    fi

    # --- Verification Phase ---
    echo "  🔍 Running Cline (Verification mode)..."
    generate_session_state "$ITERATION" "verify"

    VERIFY_MSG="IMPORTANT: Headless runner (DO NOT ask questions). First read '.cline_context/.session_state.md' to understand what has been done so far. 
    [STABILITY PROTOCOL]: Do not start by reading the entire codebase. Run the project's primary test suite immediately (check the TOOLCHAIN block in .clinerules for the correct command). Use the failures to identify which files actually need inspection.
    1) Verify all tasks from .clinerules are implemented and the code runs as expected. 
    2) [QUALITY RECONCILIATION]: Read '.cline_context/quality_audit.md'. If current implementation has resolved any of these critiques, REMOVE them from the file.
    3) MUST DO: Create a 'README.md' file that clearly explains what the project is and EXACTLY how to run it. 
    4) Check if '.cline_context/.build_issues.md' already exists. If it does, READ it. Cross off or remove the issues that were fixed in this iteration.
    5) If the app is 100% working, safe, and has a README, create a file named '.build_complete' in the root directory containing 'VERIFIED'.
    6) CONTINUITY: Watch for '[STABILITY MONITOR]' markers in history. If a turn was cut off, do not re-read from the beginning; pick up exactly where you left off.
    7) Before testing, if a port is in use, YOU MUST ONLY use 'npx kill-port <portnumber>' to free it."
    set +e
    export CLINE_BIN CLINE_MODEL VERIFY_MSG
    script -q -e -c '"$CLINE_BIN" -v --auto-approve true \
        -P openai-compatible \
        -m "$CLINE_MODEL" \
        --timeout 1800 \
        "$VERIFY_MSG"' \
        "/workspace/.cline_logs/verify_log_iter_${ITERATION}.txt"
    set -e

    # --- Safety Phase ---
    echo "  🛡️ Running Cline (Safety audit)..."
    generate_session_state "$ITERATION" "safety"

    SAFETY_MSG="IMPORTANT: Headless runner (DO NOT ask questions). First read '.cline_context/.session_state.md' to understand what has been done so far.
    [STABILITY PROTOCOL]: Do not perform an exhaustive top-to-bottom audit of every file. Use 'searchFiles' (grep) to hunt for hazardous patterns like 'unsafe', 'shell', or hardcoded paths. Only deep-dive into the specific files and lines that flag these risks.
    1) Audit for: Input validation, Path traversal, Hardcoded secrets, Injection risks, Infinite loops, Missing error handling. 
    2) If you find critical issues, attempt to FIX THEM DIRECTLY in the code. 
    3) If you fix them or the code is already safe, append 'SAFE' to the '.build_complete' file. 
    4) CONTINUITY: Watch for '[STABILITY MONITOR]' markers in history. If a turn was cut off, do not re-read from the beginning; pick up exactly where you left off.
    5) Before testing, if a port is in use, YOU MUST ONLY use 'npx kill-port <portnumber>' to free it.
    NON-INTERACTIVE MODE: Do not ask for user input. If safety issues are found, perform remediation autonomously."
    set +e
    export CLINE_BIN CLINE_MODEL SAFETY_MSG
    script -q -e -c '"$CLINE_BIN" -v --auto-approve true \
        -P openai-compatible \
        -m "$CLINE_MODEL" \
        --timeout 1800 \
        "$SAFETY_MSG"' \
        "/workspace/.cline_logs/safety_log_iter_${ITERATION}.txt"
    set -e

    # --- Check completion ---
    if [ -f "/workspace/.build_complete" ]; then
        COMPLETE_CONTENT=$(cat /workspace/.build_complete)
        if echo "$COMPLETE_CONTENT" | grep -q "VERIFIED" && echo "$COMPLETE_CONTENT" | grep -q "SAFE"; then
            echo ""
            echo "  ✅ Build VERIFIED and SAFE on iteration ${ITERATION}"
            BUILD_COMPLETE=true
        else
            echo "  ⚠ .build_complete exists but not fully verified/safe yet"
            rm -f /workspace/.build_complete
        fi
    else
        echo "  ⚠ Build not yet complete, will retry..."
    fi
done

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "========================================"
if [ "$BUILD_COMPLETE" = true ]; then
    echo "✅ BUILD PIPELINE COMPLETE"
    echo "   Iterations used: ${ITERATION}/${MAX_ITERATIONS}"
else
    echo "⚠️  BUILD PIPELINE ENDED (max iterations reached)"
    echo "   Iterations used: ${ITERATION}/${MAX_ITERATIONS}"
    echo "   Check .cline_context/.build_issues.md for remaining work"
fi

# Final size report
FINAL_SIZE=$(du -sm /workspace 2>/dev/null | cut -f1)
echo "   Final workspace size: ${FINAL_SIZE} MB"
echo "========================================"