# Dataset timing and protocol review plan

**Goal:** Explain the historical behavior, provide a mentor-facing requirements review, and export per-dataset timings from interrupted or completed v2 runs without changing their frozen evaluator or records.

**Architecture:** Add an external, read-only scripts/export_eval_v2_timing.py command. Validate the frozen manifest/selection identity and job mappings, read complete journal lines only, and reuse the existing pure timing_summary aggregation. Write a new timing snapshot directory with CSV/JSON and interpretation notes. Separate benchmark groups, MSSBench modules, generation/judge phases, complete-session stage wall, model startup and missing pipeline end events. Clearly label mixed-benchmark intervals as overlapping, not isolated dataset runtimes.

**Scope:** No model/API/SSH/Git execution by the assistant. Existing runs are diagnostic for the final experiment; the user requests a fresh formal rerun after requirements are confirmed. Do not change output budgets, concurrency, prompts, metric rules or automatically stop/resume any server task.

## Tasks

1. Review actual legacy configs and logs: HF 1024/2048 already truncated and repeated; current 65536 is 32 times 2048. Explain that backend/default randomness and sample coverage also changed; do not claim vLLM caused the behavior.
2. Add offline timing regressions for overlapping benchmarks, restart gaps, failure/retry costs, partial journal tails, expected versus valid jobs, source identity validation and no overwrite/mutation.
3. Implement timing export outside evaluation_v2, preserving old code fingerprints. Verify both new tests and existing focused tests.
4. Write docs/mentor-experiment-review-2026-10-23.md with a sendable summary, current evidence, limitations, per-dataset timing definitions, planned fresh rerun and questions for the mentor. Do not invent unobserved dataset runtimes or send a message externally.
5. Update the main experiment ledger. Supply a short inline mentor draft and links; real per-dataset times require the user to run the exporter on server journals.

## Verification completed

All 75 focused v2 tests pass, including six timing regressions. The timing CLI help check passes; evaluation_v2 source files remain unchanged. The mentor review records confirmed observations separately from unconfirmed server error details, includes a fresh formal rerun after protocol confirmation, and explains the difference between mixed-benchmark timing spans and isolated per-dataset elapsed time. No server, model, API or external-message operation was performed.
