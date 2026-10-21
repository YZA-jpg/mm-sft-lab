"""Observe one original benchmark input over SSE; preserve every attempt, no judge calls."""

import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from smoke_vllm_benchmarks import TASKS, find_data_root, make_payload, wait_for_server
from start_vllm_model import REPO, checkpoint_identity, digest, now, read_json, sha_file, write_json
from mm_sft_lab.common import stage_lock
from mm_sft_lab.data import load_prepared


def streaming_payload(payload):
    if set(payload) != {"model", "messages"}:
        raise ValueError("Expected model/messages only; generation overrides are not allowed")
    return {**payload, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}


class RemoteStreamError(ValueError):
    def __init__(self, value):
        super().__init__(json.dumps(value, ensure_ascii=False))
        self.code = value.get("code") if isinstance(value, dict) else None


def retryable_http(code):
    return type(code) is int and (code in (408, 409, 425, 429) or 500 <= code <= 599)


def is_guard(detail):
    return any(word in detail.lower() for word in ("data_inspection_failed", "content_filter", "safety_violation"))


def capture_attempt(base_url, payload, folder, idle_timeout, progress_seconds, *, opener=None):
    folder.mkdir(parents=True, exist_ok=False)
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    started, last_sync, last_progress = time.perf_counter(), time.perf_counter(), 0.0
    text, events, usage, finish, provider_id = [], 0, {}, None, None
    first_event, first_text, done = None, None, False
    result = {"status": "error", "error_kind": None, "retryable": False, "http_status": None}
    started_at = now()

    def progress():
        return {"elapsed_seconds": time.perf_counter() - started, "events": events,
            "completion_tokens_reported": usage.get("completion_tokens"),
            "text_characters": sum(map(len, text)), "tail": "".join(text[-120:])[-300:]}

    with (folder / "events.jsonl").open("x", encoding="utf-8", newline="\n") as journal, \
         (folder / "response.txt").open("x", encoding="utf-8", newline="\n") as output:
        try:
            request = urllib.request.Request(base_url + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            with opener(request, timeout=idle_timeout) as stream:
                result["http_status"] = 200
                pending = []
                for raw in stream:
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        pending.append(line[5:].lstrip(" "))
                        continue
                    if line and not line.startswith(("event:", "id:", "retry:")):
                        raise ValueError("Expected an SSE data event")
                    if line or not pending:
                        continue
                    data = "\n".join(pending)
                    pending.clear()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    events += 1
                    elapsed = time.perf_counter() - started
                    first_event = elapsed if first_event is None else first_event
                    journal.write(json.dumps({"received_at": now(), "elapsed_seconds": elapsed, "chunk": chunk}, ensure_ascii=False) + "\n")
                    if "error" in chunk:
                        raise RemoteStreamError(chunk["error"])
                    incoming_id = chunk.get("id")
                    if incoming_id:
                        if provider_id and provider_id != incoming_id:
                            raise ValueError("Response identity changed inside one stream")
                        provider_id = incoming_id
                    if chunk.get("usage"):
                        if not isinstance(chunk["usage"], dict):
                            raise ValueError("Invalid stream usage")
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise ValueError("Expected one completion per request")
                        delta = choice.get("delta", {}).get("content")
                        if delta is not None:
                            if not isinstance(delta, str):
                                raise ValueError("Invalid content delta")
                            if delta:
                                first_text = elapsed if first_text is None else first_text
                                text.append(delta)
                                output.write(delta)
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                    tick = time.perf_counter()
                    if tick - last_sync >= 1:
                        for stream_file in (journal, output):
                            stream_file.flush()
                            os.fsync(stream_file.fileno())
                        last_sync = tick
                    if elapsed - last_progress >= progress_seconds:
                        state = progress()
                        write_json(folder / "progress.json", state)
                        print(f"{folder.name}: {elapsed:.1f}s; tokens={state['completion_tokens_reported']}; chars={state['text_characters']}", flush=True)
                        last_progress = elapsed
                if not done or finish is None:
                    raise http.client.IncompleteRead(b"")
                if finish == "content_filter":
                    result.update(error_kind="provider_guardrail")
                else:
                    result.update(status="ok")
        except (urllib.error.HTTPError, RemoteStreamError) as error:
            detail = error.read().decode("utf-8", errors="replace")[:2000] if isinstance(error, urllib.error.HTTPError) else str(error)
            guard = is_guard(detail)
            result.update(error_kind="provider_guardrail" if guard else "http_error",
                retryable=not guard and retryable_http(error.code), http_status=error.code,
                exception_class=type(error).__name__, error=detail)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.IncompleteRead) as error:
            result.update(error_kind="network_error", retryable=True, exception_class=type(error).__name__,
                reason_class=type(error.reason).__name__ if isinstance(error, urllib.error.URLError) else None,
                error=f"{type(error).__name__}: {error}")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            result.update(error_kind="format_error", exception_class=type(error).__name__, error=str(error))
        except KeyboardInterrupt:
            result.update(status="interrupted", error_kind="manual_interrupt")
        finally:
            for stream_file in (journal, output):
                stream_file.flush()
                os.fsync(stream_file.fileno())
    response = "".join(text)
    result.update(started_at=started_at, finished_at=now(), wall_seconds=time.perf_counter() - started,
        first_event_seconds=first_event, first_text_seconds=first_text,
        finish_reason=finish, done_received=done, usage=usage, provider_response_id=provider_id,
        stream_events=events, response_characters=len(response), response_sha256=digest(response),
        partial_output=result["status"] != "ok" and bool(response),
        issue_counts={"truncated": 1} if finish == "length" else {"empty": 1} if not response and result["status"] == "ok" else {})
    write_json(folder / "progress.json", progress())
    write_json(folder / "summary.json", result)
    return result


def capture_with_retries(base_url, payload, folder, idle_timeout, progress_seconds, *, opener=None, sleeper=None):
    attempts, sleeper = [], sleeper or time.sleep
    for index in range(6):
        result = capture_attempt(base_url, payload, folder / f"attempt-{index + 1:02d}",
                                 idle_timeout, progress_seconds, opener=opener)
        attempts.append(result)
        write_json(folder / "attempts.json", attempts)
        if not result["retryable"] or index == 5:
            break
        delay = 2 ** index
        attempts[-1]["backoff_seconds"] = delay
        write_json(folder / "attempts.json", attempts)
        sleeper(delay)
    return {"status": result["status"], "attempts": attempts, "successful_attempt": len(attempts) if result["status"] == "ok" else None,
        "earlier_partial_attempts": sum(row["partial_output"] for row in attempts[:-1]),
        "scope": "Streaming diagnostic only; no judge or ASR labels"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--id", default="mssbench/embodied/0000/1/a")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--idle-timeout-seconds", type=int, default=600)
    parser.add_argument("--progress-seconds", type=int, default=10)
    args = parser.parse_args(argv)
    if min(args.idle_timeout_seconds, args.progress_seconds) <= 0:
        parser.error("Idle timeout and progress interval must be positive")
    root = find_data_root(args.data_root)
    records, _, data = load_prepared(root, {"require_locked_data": True, "tasks": TASKS, "per_task_limit": None})
    selected = [row for row in records if row["id"] == args.id]
    if len(selected) != 1 or data.get("is_mock"):
        raise ValueError("Expected exactly one real validated request")
    server = read_json(args.server_dir / "server.json")
    if digest(server["settings"]) != server["signature"] or checkpoint_identity(server["model_path"]) != server["settings"]["model"]:
        raise ValueError("Server receipt or checkpoint changed")
    payload = streaming_payload(make_payload(selected[0], root, data, server["model_id"]))
    output = (args.output or REPO / "runs" / ("vllm-stream-probe-" + now().replace(":", "-") + "-" + uuid.uuid4().hex[:6])).resolve()
    with stage_lock(output, "stream-probe"):
        if any(p.name != ".stream-probe.lock" for p in output.iterdir()):
            raise ValueError("Use a fresh diagnostic directory")
        manifest = {"schema_version": 1, "status": "initialized", "created_at": now(), "id": args.id,
            "request_sha256": digest(payload), "server_signature": server["signature"], "launch_id": server["launch_id"],
            "data_manifest_sha256": sha_file(root / "prepared/manifest.json"), "script_sha256": sha_file(__file__),
            "transport_options": {key: payload[key] for key in ("stream", "stream_options")},
            "idle_timeout_seconds": args.idle_timeout_seconds, "total_duration_limit": None,
            "generation_parameters_sent": [], "scope": "One-request streaming diagnostic; no judge or benchmark score"}
        write_json(output / "manifest.json", manifest)
        write_json(output / "selection.json", selected[0])
        print("Streaming output:", output, flush=True)
        try:
            wait_for_server(server, 900)
            write_json(output / "manifest.json", {**manifest, "status": "running"})
            result = capture_with_retries(server["base_url"], payload, output, args.idle_timeout_seconds, args.progress_seconds)
            write_json(output / "summary.json", result)
            write_json(output / "manifest.json", {**manifest, "status": result["status"], "updated_at": now()})
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return 130 if result["status"] == "interrupted" else 0 if result["status"] == "ok" else 2
        except BaseException:
            write_json(output / "manifest.json", {**manifest, "status": "interrupted", "updated_at": now()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
