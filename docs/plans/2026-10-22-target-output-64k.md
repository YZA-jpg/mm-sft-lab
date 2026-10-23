# Target output 65,536 tokens implementation plan

**Goal:** Apply the user's explicit output budget to target generation only and repeat a bounded, concurrent SIUO/MSSBench test without reusing the interrupted unlimited run.

**Architecture:** Add an optional `generation.max_tokens` protocol field, defaulting to null for the existing omission policy. The runner forwards it only for the generation phase. The HTTP client accepts this one explicit decoding control; judge requests continue to omit all decoding controls. Frozen signatures and cross-model comparison identities include the budget. Keep the existing scheduler, network retry budget, stream capture and owned vLLM lifecycle.

**Tech stack:** Python standard library, existing v2 data/HTTP/SSE runner, local synthetic tests. The user executes all Git, server, GPU and API commands.

## Implementation and verification

1. Add offline tests for positive-integer validation, exact target JSON/SSE bodies, no judge leakage, length termination without retries, cap-bound resume/comparison, and simultaneous four-request scheduling. Tests use synthetic model output and no network.
2. Extend config, client and generation forwarding. Reflect the explicit cap in plan, generation manifests/status, metrics and sweep metadata; retain truncation counts/rates by benchmark and all retry wall time.
3. Add two new configs: `eval_v2.siuo_mss_64k_smoke.json` and `eval_v2.siuo_mss_64k_full.json`, both cap 65,536, streaming and concurrency four. The smoke limit selects the first three units per split, retaining MSS pairs: 9 SIUO + 6 chat + 6 embodied = 21 jobs on the existing real snapshot. Verify that both known long IDs are selected. Keep previous no-explicit-cap configs unchanged.
4. Run focused v2 offline tests and plan the real local snapshot without inference. Record the scope change in the data/experiment notes and source-adaptation notes. Provide selective local Git commands followed by a detached, generation-only managed server run on GPU 4/port 18004, fixed KV 40 GiB, with startup/cleanup/GPU samples and per-benchmark timing retained.

Also provide `configs/eval_v2.64k_full.json` for the future eight-benchmark run; its presence is not evidence that the remaining real datasets/WildGuard are ready. All previous no-explicit-cap config files stay unchanged. Existing evaluation snapshots are reused without re-preparation or new training-overlap checks.

## Validation and user execution

The eight initial budget regressions passed after adding target-only forwarding; the existing 51 focused v2 tests also passed. The final suite additionally exercises cap preservation across a retry with only attempts 5/6 remaining: **60 focused offline tests pass**. Local real-data plans verify 21 smoke and 2,461 full generation jobs and both known long IDs in each selection. No SSH, real API/model request, GPU execution or Git commit/push is performed by the assistant.

Use the managed CLI through `scripts/run_eval_v2.sh` with `--generate-only`, `--gpu 4`, `--vllm-python "$PWD/.venv-vllm/bin/python"`, `--gpu-memory-utilization 0.65` and `--target-kv-cache-gib 40`. Start that entire command with nohup and redirected stdin/stdout/stderr, record its PID and a new run-directory pointer before returning. It must not resume the old no-explicit-cap directory. Verify that previous generation/managed processes are stopped, and use the legacy launcher's identity-checked stop command for its own old service before starting this managed instance. Inspect `job.log`, `generation/manifest.json`, `report/metrics.json`, `report/timing.csv` and `gpu-samples.jsonl`. A successful generation-only exit is not a semantic judge score.

## Interpretation

- `max_tokens` limits newly generated tokens, not prompt+output context. No temperature, seed, top_p, top_k, repetition penalty or custom early-stop rule is added.
- A length-limited answer is retained as a generated answer and separately counted as truncated. It is not a transport failure and is not retried solely for reaching the budget.
- All baseline/SFT models in this protocol must use the same cap; old unlimited outputs are not imported into the new run. Other protocol choices remain provisional.
- Earlier service logs report a 291,264-token KV pool. Four full output budgets total 262,144 tokens before inputs; this motivates concurrency four, not a guarantee of available memory or absence of scheduler preemption on a shared GPU. Inspect fresh GPU availability before startup.
- This change bounds a repeated answer; it does not fix the model's observed repetition or establish a runtime promise. A 65,536-token answer can still be expensive.

## Full SIUO/MSSBench launch preparation

The user authorized preparing the full run after the GPU 1 smoke. Rechecked the existing real local snapshot: 167 SIUO samples with trials 0/1/2 (501 jobs), 1200 chat and 760 embodied MSSBench samples with trial 0 (1960 jobs), 2461 unique job IDs total. Safe/unsafe each have 980 MSSBench requests. All 60 focused v2 offline tests pass again. Existing code/config already provide the required repeats, durable results/attempts, exclusion quality metrics and per-benchmark timing; no behavior change is needed.

The user-run server instructions are in ../siuo-mss-64k-full-run.md. They verify the server's 21/21 smoke report and full plan, check current GPU 1 memory, save plan/commit/PID/pointer, and submit the entire generation-only managed lifecycle to nohup in a new full directory. The earlier GPU 4 startup failed for insufficient memory; the fresh GPU 1 check is not a reservation or guarantee. Judges follow later in the same run, one logical judgment per valid generated response; network retries never add ASR trials. Other six benchmarks are outside this launch. No remote execution or final full score is claimed.
