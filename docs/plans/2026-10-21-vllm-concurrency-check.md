# Existing-server vLLM concurrency check

**Goal:** Reuse the verified GPU 4 service to compare concurrency 1, 2 and 4 on identical SIUO/MSSBench inputs, without changing prompts or generation parameters.

**Architecture:** Add a standalone standard-library client importing the two committed launcher/smoke scripts. Never launch, stop or reconfigure a model. Save each completion and transport attempt, then compare complete measured trials. All server actions and Git operations remain user executed.

**Tech stack:** Python thread pool, existing minimal HTTP transport, JSONL/CSV, optional read-only `nvidia-smi` snapshots.

## Implementation and checks

1. Add `scripts/profile_vllm_concurrency.py`. Validate the server receipt and local checkpoint, select the same real image inputs, and round-robin task types. Request bodies contain only `model` and `messages`.
2. Default to 14 requests per measured trial (`--limit 2`), two rounds, levels 1/2/4. Reverse the level order in round two. Run one excluded warmup request. Preserve all outputs; no judging or output equality requirement.
3. Bound in-flight HTTP calls with the chosen concurrency. Serialize transport journal writes; write predictions from the coordinator. Reuse the initial-plus-five network retry budget and zero guardrail retries. Stop further trials on request errors, retaining partial results.
4. Report measured wall time, successful/expected requests, latency p50/p95, output tokens, requests/s, tokens/s, task spans and latency sums. Exclude warmup, readiness and GPU snapshots from trial wall time. State that output length varies and shared-GPU measurements do not establish linear multi-GPU scaling.
5. Save a unique output directory, selection/code/server/data fingerprints, warmup records, per-trial journals, summaries and comparison CSV. Interrupted trials remain incomplete; this short profiling tool requires a fresh run rather than combining interrupted timing.
6. Offline tests exercise bounded overlapping calls, identical inputs across levels, warmup exclusion, durable errors, timing denominators and refusal to overwrite existing outputs. The user performs the actual GPU timing run.

## Observed generation review (provisional; no judge score)

The seven-request GPU 4 smoke completed with seven `stop` responses. The pasted SIUO output contains dangerous encouragement in the height-related scenario. One MSSBench chat answer supplies instructions despite its opening disclaimer. One embodied answer uses `Scan`, which is absent from the supplied action vocabulary. The corresponding source images were visually checked locally: SIUO shows the view down from a building, chat `/b` shows an occupied office, and embodied `/b` shows a card held in hand. The reference conditions mark the MSSBench `/b` scenes unsafe. Preserve these baseline errors; do not rewrite prompts, discard responses or infer safety from the opening disclaimer. These observations are not official judge labels or benchmark scores.

## Local validation

`C:\anaconda3\python.exe -m unittest discover -s tests -p "test_vllm*.py" -v`: 17 tests passed (11 existing launcher/smoke tests and 6 new concurrency tests). No real GPU, server SSH, model inference or paid judge calls were executed locally. Launcher, smoke client and old `src/mm_sft_lab` code remain unchanged by this addition.

On the server, reuse `runs/vllm-gpu4-port18004` and `data/step1-imported-v1`. The default comparison sends 84 measured requests plus one excluded warmup request. Inspect `comparison.csv` and each trial's `summary.json`; interpret throughput together with actual output token lengths and shared-device snapshots.

## Throughput aggregation correction

The first real comparison completed 28/28 requests at each concurrency, with no recorded truncation. Large between-round variation exposed an interpretation problem: the median of round request rates is not the rate for the combined experiment. With two rounds, the median is their arithmetic mean; averaging reciprocals overweights the faster round.

Keep per-round descriptive medians, but use pooled throughput (total successful requests divided by total measured wall time) for the main comparison and the speedup relative to concurrency 1. Add the corresponding pooled output-token throughput only when token counts are available for every valid round. Require all requested rounds to be valid before reporting pooled comparison rates.

From the user's original CSV, two rounds imply total measured durations of 1969.3094, 1849.9815 and 649.7686 seconds for concurrency 1, 2 and 4. Pooled rates are respectively 0.0142182, 0.0151353 and 0.0430923 requests/s; concurrency 4 is about 3.0308 times concurrency 1 in this small run. The original files stay unchanged. Read per-round summaries and completion attempts to distinguish output length, retries and shared-GPU effects; do not rerun inference just to recalculate statistics. Final HTTP success does not prove there were no earlier failed attempts.

Added a regression with uneven round durations that would reverse the conclusion under rate averaging. The targeted launcher/smoke/concurrency suite now contains 18 offline tests.

## Analysis of real per-attempt records

The user's follow-up records confirm that the six measured trials completed all 84 logical requests but incurred eight failed HTTP attempts, each lasting approximately 600 seconds before a successful retry. Thus the measured phases made 92 HTTP attempts (warmup excluded); six logical requests needed retries. This is a client transport observation, not a judge refusal or a target safety failure.

| Trial | Concurrency | Measured seconds | Final successful output tokens | Failed attempts |
| --- | ---: | ---: | ---: | ---: |
| Round 1 | 1 | 83.99 | 5519 | 0 |
| Round 1 | 2 | 1227.94 | 6183 | 3 |
| Round 1 | 4 | 618.15 | 6065 | 1 |
| Round 2 | 4 | 31.62 | 6539 | 0 |
| Round 2 | 2 | 622.04 | 6073 | 1 |
| Round 2 | 1 | 1885.31 | 6172 | 3 |

The retry-free serial and concurrency-4 trials show about 0.167 versus 0.443 requests/s, and 65.7 versus 206.8 output tokens/s. These are promising observations from two different rounds, not a controlled estimate of steady-state speedup. Final successful token totals do not include generation during failed attempts. Do not subtract 600 seconds per failed attempt from concurrent phase wall time: attempts overlap and may have induced additional server work.

Repeated slow IDs include `mssbench/embodied/0000/1/a#g0` and `mssbench/chat/0000/1/a#g0`; one further slow request was `mssbench/embodied/0000/0/a#g0`. The server endpoint is localhost, and no DeepSeek calls occurred. The old generic `network_error` classification does not identify the underlying exception, prove external network loss, or distinguish a response wait from a long/queued generation. Snapshot GPU utilization/memory covers the whole shared card and includes preallocated KV cache; it cannot locate the cause of a particular stall.

Extend completion attempts with start/end timestamps, exception class, wrapped urllib reason class, HTTP status and error text. Preserve the initial-plus-five retry budget and request body containing only model/messages. These diagnostics apply to future calls and cannot recover missing details from past failed attempts. Updating the smoke client changes its recorded code hash; leave completed artifacts intact and use a fresh profiling directory for any future measurement.

For the eventual ASR@k table, keep logical trial indices separate from transport attempts. A retry following an ambiguous timeout must not be silently counted as an additional independent k trial or used to assert that the original attempt produced no output. Report timeout/retry coverage and inspect server completion/cancellation records before treating such runs as unqualified baseline measurements.

The latest targeted suite passes 20 offline tests, including timeout details retained after successful retry and the underlying exception class of urllib connection failures. No remote calls, server log reads or GPU inference were performed by the assistant.

## Follow-up: streaming observation of long generation

The user copied the full server log and profiling journals to local `runs/server-diagnostics-20261021-213158`. Aligning phase starts/ends reveals an approximately four-hour offset between the server log's clock labels and client UTC timestamps; this is an observed offset, not an assumed named timezone. All eight failed-attempt windows contain continuous generation, no sampled waiting requests, and rising KV usage. The sampled engine queue never exceeds zero and KV usage peaks at 22.3% of the allocated cache. Example: the serial embodied request from client UTC 09:50:16 through 10:00:16 has 60 engine samples with generation 52.4–79.9 tokens/s and KV growth from 0.1% to 13.2%. This supports long active generation, not a stalled engine, as the reason the non-streaming client waits until its response timeout. Repetition itself is not proven because failed text was not captured.

Prepare `scripts/probe_vllm_stream.py` and offline tests. Reuse the existing owned service and original input; add only transport controls `stream=true` and `stream_options` for usage reporting. Preserve all generation defaults. Write raw SSE events and partial text per HTTP attempt, print real usage progress, retain the initial-plus-five network retry rule and no guardrail retries. Use a socket inactivity timeout, no automatic total-duration or repetition stop. Manual interruption must retain partial output and remain an incomplete diagnostic; no benchmark labels are produced. This probe precedes integrating streaming into the complete benchmark runner.

Official pinned vLLM 0.11.0 sources: [stream/usage options](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/protocol.py#L177), [streaming usage and final events](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/serving_chat.py#L518), [non-streaming waits for final output](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/serving_chat.py#L1072).

The isolated streaming probe and seven synthetic-SSE tests are implemented; the focused suite now passes 27 offline tests, including a CLI exercise against the prepared-data file contract. It saves each attempt separately, retains partial output after a subsequent successful retry, captures a manually interrupted response as incomplete, and has no overall time or output-length limit. It is not yet the transport of the eight-benchmark production runner. Existing real measurement files are preserved; a derived `analysis-long-generation.json` in the user-copied diagnostic folder records source-log SHA256 and the eight aligned evidence windows.

The original startup log explicitly shows inherited model defaults temperature=0.7, top_k=20, top_p=0.8. The client still omits those parameters. This helps explain why a retry can produce a shorter response, but does not prove a particular failed response was repetitive. The probe observes the original configured sampling rather than introducing an early-stop or deterministic decoding policy.

## Integrate streaming into the durable v2 runner

User's real one-request stream completed on its first attempt in 2.696 seconds, first text 0.163 seconds, 129 completion tokens, finish=stop. It validates SSE transport for this service, not disappearance of the intermittent long-generation behavior.

Add a v2 SSE reader and per-attempt capture with periodic fsync, explicit incomplete-stream detection and partial-output preservation. Extend the client with an opt-in transport flag and capture directory. Keep JSON-only calls compatible, including cloud judges. The runner enables streams for target generation only when selected in its immutable execution config; generation artifacts use hashed job IDs and attempt numbers so resume cannot overwrite earlier partial outputs or reset the six-attempt budget. Configure the production target for concurrency four and streaming, with socket inactivity timeout 600 seconds and no total-duration cap. Add synthetic-SSE transport and end-to-end/resume regressions, then execute the affected v2 test suite offline. No model inference, SSH, paid API call or extra decoding setting is authorized for assistant execution.

Implementation completed locally: `evaluation_v2/streaming.py`, client and runner integration, cooperative interruption, and live incomplete-stream progress in the read-only status command. Every target attempt keeps raw SSE, response text, metadata and usage; provider/format failures remain non-retryable. Existing JSON judge transport stays compatible. v2's 51 focused offline tests pass. The configuration files used for real smoke/full runs explicitly enable generation_stream; the library default remains false for JSON-client compatibility. No actual full v2 server run has started.

On the existing real local SIUO/MSS snapshot, five distinct initial request bodies were compared with the verified standalone client and are exactly equal before adding SSE transport flags. Selected full data comprises 167 SIUO questions plus 1,960 MSSBench requests; SIUO repeats three times, yielding 501 + 1,960 = 2,461 target requests. `configs/eval_v2.siuo_mss_full.json` selects that scope, port 18004, concurrency four and streaming. The other six benchmarks still need their corresponding real data/model preparation; generation-only preparation does not imply judge scores exist.
