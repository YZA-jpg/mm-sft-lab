# WebP MIME compatibility recovery plan

**Goal:** Repair the locally reproduced missing-WebP-MIME request preparation error and allow the user's interrupted full run to resume without changing its frozen evaluation code or rerunning successful generations.

**Architecture:** Add a separate scripts/resume_eval_v2_webp.py entry point. It reads and validates the original run, accepts only the specific NoneType/str concatenation preparation error on WebP input, registers image/webp in the Python process, and checks every already-successful request hash against reconstructed requests. The original evaluation_v2/*.py files remain byte-identical. Save an external repair receipt and resolved frozen config, then delegate --resume --generate-only to the original CLI with its recorded model/runtime/GPU settings.

**Scope:** Local code/tests/documentation only. The user performs all Git, server, GPU and API operations. The server's exact error is not yet supplied; the entry point must refuse unrelated errors, including disk/network failures. Do not lower max_tokens or change concurrency, retry budgets, prompts, source images or trial counts.

## Steps

1. Reproduce the exact failing local sample mssbench/chat/0031/0/b: its .webp MIME guess is None, causing TypeError before HTTP. Inventory the real snapshot for other missing MIME suffixes.
2. Add offline regressions for missing WebP MIME, original bytes and request identity preservation, skipping successful results on resume, retaining the failed history, refusing unrelated failures/live locks, and check-only mode with no GPU/API execution.
3. Implement the external compatibility entry point, including an audit receipt and frozen-argument reconstruction; never rewrite run journals, fingerprints or evaluation_v2 code.
4. Run the focused regressions and the existing v2 suite. Document the early-stop timestamp, observed truncation burden and user-run check/launch commands.
5. Have the user run --check-only first. A matching server failure and unchanged successful requests permit a detached resume of the original directory; otherwise inspect the reported actual error before proceeding.

## Observed evidence

At the user's 2026-10-23 12:07 Beijing snapshot, 500 generations are valid, one local preparation error is recorded, and 1960 are unattempted. The last completion timestamp maps to 2026-10-23 01:07 Beijing, about 11 hours earlier. Of the valid responses, 46 reached the output budget (9.2%); at the declared 65536 cap they account for 3,014,656 final-output tokens. This does not yet include all normal answers or failed HTTP attempts. Do not infer the server's full error solely from its broad local_preparation_error classification.

## Completed local verification

69 focused v2 tests pass, including nine new recovery regressions. All 2127 real local request bodies (2461 logical generation jobs) serialize after registering image/webp; the only unknown-MIME inputs before registration were five WebP assets. Frozen evaluation_v2 files remain unchanged. The original server error remains unconfirmed until the user's guarded preflight reads it. User-run publication, checking and detached-resume instructions are recorded in ../webp-preparation-incident.md; no remote execution has been performed.
