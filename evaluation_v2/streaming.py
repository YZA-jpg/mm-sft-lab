"""SSE assembly and durable per-attempt output; no decoding settings or overall deadline."""

import http.client
import json
import os
import time
from pathlib import Path

from .common import digest, now, write_json


class StreamRemoteError(ValueError):
    def __init__(self, error):
        super().__init__(json.dumps({"error": error}, ensure_ascii=False))
        self.code = error.get("code", 500) if isinstance(error, dict) else 500


class StreamJournalError(RuntimeError):
    pass


class Capture:
    def __init__(self, directory, number, secret=""):
        self.directory = Path(directory) / f"attempt-{number:02d}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.secret, self.journal, self.output = secret, None, None
        try:
            self.journal = (self.directory / "events.jsonl").open("x", encoding="utf-8", newline="\n")
            self.output = (self.directory / "response.txt").open("x", encoding="utf-8", newline="\n")
        except OSError as error:
            if self.journal:
                self.journal.close()
            raise StreamJournalError(str(error)) from error
        self.synced = time.perf_counter()

    def record(self, chunk, state, delta):
        try:
            line = json.dumps({"received_at": now(), "elapsed_seconds": time.perf_counter() - state["tick"],
                               "chunk": chunk}, ensure_ascii=False)
            self.journal.write((line.replace(self.secret, "[REDACTED]") if self.secret else line) + "\n")
            self.output.write(delta)
            tick = time.perf_counter()
            if tick - self.synced >= 1:
                self.sync()
                write_json(self.directory / "progress.json", self.progress(state))
                self.synced = tick
        except OSError as error:
            raise StreamJournalError(str(error)) from error

    @staticmethod
    def progress(state):
        return {"updated_at": now(), "elapsed_seconds": time.perf_counter() - state["tick"],
            "events": state["events"], "text_characters": state["characters"],
            "completion_tokens_reported": state["usage"].get("completion_tokens"),
            "finish_reason": state["finish_reason"], "done_received": state["done"]}

    def sync(self):
        for stream in (self.journal, self.output):
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, state, result):
        try:
            self.sync()
            metadata = {**self.progress(state), "status": result["status"], "error_kind": result["error_kind"],
                "response_sha256": digest("".join(state["parts"])),
                "partial_output": result["status"] != "ok" and bool(state["characters"])}
            write_json(self.directory / "summary.json", metadata)
            return {"directory": str(self.directory), **metadata}
        except OSError as error:
            raise StreamJournalError(str(error)) from error
        finally:
            self.journal.close()
            self.output.close()


def new_state():
    return {"tick": time.perf_counter(), "events": 0, "parts": [], "refusal": [], "usage": {},
        "characters": 0, "finish_reason": None, "done": False, "id": None, "model": None,
        "first_event_seconds": None, "first_text_seconds": None}


def read_stream(response, state, *, chat, capture=None, cancel_event=None):
    pending = []
    for raw in response:
        if cancel_event is not None and cancel_event.is_set():
            raise KeyboardInterrupt
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            pending.append(line[5:].lstrip(" "))
            continue
        if line and not line.startswith(("event:", "id:", "retry:")):
            raise ValueError("Expected SSE data events")
        if line or not pending:
            continue
        data = "\n".join(pending)
        pending.clear()
        if data == "[DONE]":
            state["done"] = True
            break
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise ValueError("Invalid SSE envelope")
        state["events"] += 1
        if state["first_event_seconds"] is None:
            state["first_event_seconds"] = time.perf_counter() - state["tick"]
        if chunk.get("id"):
            if state["id"] and state["id"] != chunk["id"]:
                raise ValueError("SSE response identity changed")
            state["id"] = chunk["id"]
        state["model"] = chunk.get("model", state["model"])
        if chunk.get("usage"):
            if not isinstance(chunk["usage"], dict):
                raise ValueError("Invalid SSE usage")
            state["usage"] = chunk["usage"]
        deltas = []
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Expected one completion")
            delta = choice.get("delta", {}) if chat else {"content": choice.get("text")}
            text = delta.get("content")
            if text is not None:
                if not isinstance(text, str):
                    raise ValueError("Invalid SSE text delta")
                deltas.append(text)
            if isinstance(delta.get("refusal"), str):
                state["refusal"].append(delta["refusal"])
            if choice.get("finish_reason"):
                state["finish_reason"] = choice["finish_reason"]
        text = "".join(deltas)
        if text:
            state["parts"].append(text)
            state["characters"] += len(text)
            if state["first_text_seconds"] is None:
                state["first_text_seconds"] = time.perf_counter() - state["tick"]
        if capture:
            capture.record(chunk, state, text)
        if "error" in chunk:
            raise StreamRemoteError(chunk["error"])
    if not state["done"] or state["finish_reason"] is None:
        raise http.client.IncompleteRead(b"")
    text, refusal = "".join(state["parts"]), "".join(state["refusal"])
    message = {"content": text or (None if refusal else ""), "refusal": refusal or None}
    return {"id": state["id"], "model": state["model"], "usage": state["usage"],
        "choices": [{"message": message, "text": text, "finish_reason": state["finish_reason"]}]}
