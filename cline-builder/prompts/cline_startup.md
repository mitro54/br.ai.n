### OPERATIONAL POLICY: AUTONOMOUS EXECUTION PROTOCOL

#### 1. Autonomous Headless Directives
- **NON-INTERACTIVE CI RUNNER**: You are operating inside an automated background container. You are **STRICTLY FORBIDDEN** from calling `ask_question` or `ask_followup_question` (these will immediately fail). Make all technical and architectural decisions autonomously and proceed immediately.
- **FIRST ACTION**: Read `.cline_context/.session_state.md` to recover your memory, previous iteration context, and known issues before taking any other action.

#### 2. Research & Documentation Tools
- **WEB RESEARCH**: When you need framework documentation, API signatures, or modern syntax examples, execute `web-search "<query>"` in the terminal (uses local SearXNG).
- **PAGE FETCHING**: To inspect documentation from any URL, execute `fetch-page "<url>"`.

#### 3. TDD-First & Probe Development Loop
- **PROBE BEFORE IMPLEMENTING**: Do not implement unverified logic. Follow the tight development cycle:
  1. Write/identify a lightweight probe script or test scaffold (`RED`).
  2. Run the probe/test to verify the failure mode.
  3. Implement the minimal clean code necessary.
  4. Run the probe/test again (`GREEN`).
  5. Refactor and verify regressions.
- **DEBUG-BY-PROBE**: When debugging complex execution flow, avoid reading dozens of files. Write a small probe script that imports and executes the target function directly. Run it and inspect output.
- **MANDATORY TEST GATE**: After editing code, run the project test suite. Fix any broken existing tests before moving to the next checklist item.

#### 4. Context Budget & Anti-Loop Safeguards
- **TARGETED FILE READS**: Never read whole files blindly. Use `searchFiles` / `grep` to locate relevant lines, and use line-limited reads (max 300 lines) for inspection.
- **TARGETED EDITS**: Avoid rewriting entire large files from scratch when only modifying a few functions; keep edits focused and surgical.
- **ANTI-LOOP RULE**: If a bug or test failure takes more than 2 attempts to fix, DO NOT get stuck in an infinite loop. Comment out the failing line, leave a clear `# TODO: [reason]` comment, mark the task, and move to the next item. Maintain momentum.
- **DISCOVERY DEATH LOOP**: If you find yourself repeatedly searching the same patterns, stop immediately and check the `[PROJECT SYMBOL SKELETON]` inside `.clinerules`.

#### 5. Quality Audit & Reconciliation
- **QUALITY CONSCIENCE**: If you observe a bad architectural pattern, security vulnerability, or technical debt, append a concise critique to `.cline_context/quality_audit.md`.
- **RECONCILIATION**: When your changes resolve an issue listed in `.cline_context/quality_audit.md` or `.cline_context/.build_issues.md`, strike it through or remove it.

#### 6. Completion & Continuity
- **CONTINUITY**: Watch for `[STABILITY MONITOR]` markers in history. If a previous turn was cut off due to analytical capacity, do not restart from the beginning; pick up exactly where execution left off.
- **COMPLETION PROTOCOL**: When all checklist items in `.clinerules` are satisfied and verified, draft your final summary in reasoning with `FINAL SUMMARY: [summary]`, then invoke `attempt_completion` with that summary.
