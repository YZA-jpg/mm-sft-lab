# Fresh baseline validation implementation plan

**Goal:** Verify the WebP request fix in a fresh smoke, then allow a separate full run with isolated benchmark timing.

**Architecture:** Use the existing v2 runner with an explicit image MIME map, offline request preflight, audited sample selection and a drain barrier between benchmarks. Keep four concurrent requests inside each benchmark. New runs get new code/config identities; never rewrite or merge old results.

**Tech stack:** Python standard library, Pillow, unittest; user-operated Linux vLLM and Bash.

1. Add regressions for missing system WebP MIME, unknown formats, selected MSS pairs, serial benchmark barriers, retry/length behavior and durable timing.
2. Fix MIME lookup and add complete request construction preflight to plan/initialize before any model load. Preserve source bytes and hashes.
3. Add explicit sample-ID selection that expands MSS pairs, plus serial benchmark scheduling with boundary journals. Preserve mixed scheduling for historical configurations.
4. Add fresh smoke/full configs: SIUO x3, MSS x1, target max_tokens=65536, concurrency=4, judges unchanged. Smoke covers all five real WebP assets including the failed request.
5. Add detached launch/status commands with explicit smoke acceptance before a separate full launch. Preserve old runs as diagnostic records.
6. Consolidate current documentation into a runbook and index; retain historical evidence and training/data provenance, without deleting unrelated files.
7. Run focused regression tests, all v2 tests, local full-data request preflight and shell syntax checks. No real network, SSH, API, GPU execution or Git publication by the assistant.

**Acceptance:** Full-data preflight constructs all 2127 sample requests (2461 jobs); fresh smoke includes five WebP assets and intact pairs; benchmark intervals do not overlap; full launch remains blocked until matching smoke is accepted.

## Local verification

Implemented the explicit MIME map, preflight, pair-preserving sample selection, serial drain barrier and boundary timing, detached launcher and full-run smoke acceptance. All 307 repository unit tests pass (87 v2 tests included). Four Bash blocks in the current runbook pass syntax checks; local Markdown links resolve. Real local preflight validates 13 samples / 15 smoke jobs and 2127 samples / 2461 full jobs; all five WebP assets are covered. The original 65536 budget and judge omissions are unchanged. No remote launch, GPU work, API calls or Git publication was performed. Server smoke remains the next user-executed validation stage.
