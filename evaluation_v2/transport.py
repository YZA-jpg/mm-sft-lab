"""OAI-compatible HTTP with no SDK hidden retries and no generation parameters."""

import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .common import digest, now
from .streaming import Capture, StreamRemoteError, new_state, read_stream

GUARD_CODES = {"data_inspection_failed", "content_filter", "content_filter_error", "responsibleaipolicyviolation", "safety_violation"}


def classify_http(status, body):
    try:
        error = json.loads(body).get("error", {})
        if not isinstance(error, dict):
            error = {}
    except (ValueError, AttributeError):
        error = {}
    code, kind = str(error.get("code", "")).lower(), str(error.get("type", "")).lower()
    message = str(error.get("message", ""))
    if code in GUARD_CODES or kind in GUARD_CODES:
        return "provider_guardrail", False
    if status == 400 and ("Multimodal file size is too large" in message or code == "context_length_exceeded"
                          or "maximum context length" in message.lower()):
        return "provider_input_limit", False
    if status in (401, 403, 404):
        return "authentication_or_configuration_error", False
    if status in (408, 409, 425, 429) or 500 <= status <= 599:
        return "network_error", True
    return "http_error", False


def retry_after(value):
    try:
        return min(300.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        try:
            return min(300.0, max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()))
        except (TypeError, ValueError, OverflowError):
            return 0.0


class Client:
    def __init__(self, endpoint, execution, *, opener=None, sleeper=None):
        self.endpoint = endpoint
        self.execution = execution
        self.opener = opener or urllib.request.urlopen
        self.sleeper = sleeper or time.sleep
        self.key = os.environ.get(endpoint["api_key_env"], "").strip() if endpoint["api_key_env"] else ""
        if endpoint["api_key_env"] and not self.key:
            raise ValueError(f"Missing {endpoint['api_key_env']} in current process environment")

    def complete(self, *, messages=None, prompt=None, attempt_offset=0, on_attempt=None, stream=False, stream_directory=None, cancel_event=None):
        if (messages is None) == (prompt is None):
            raise ValueError("Provide exactly messages or prompt")
        if type(attempt_offset) is not int or attempt_offset not in range(7):
            raise ValueError("Invalid transport attempt budget")
        payload = {"model": self.endpoint["model"], **({"messages": messages} if messages is not None else {"prompt": prompt})}
        if type(stream) is not bool or (stream_directory is not None and not stream):
            raise ValueError("Invalid stream transport configuration")
        if stream:
            payload.update(stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True})
        # Streaming/usage flags are transport controls, never sampling overrides.
        endpoint = "chat/completions" if messages is not None else "completions"
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        attempts, captures, started, total_backoff = [], [], time.time(), 0.0
        result = {"status": "error", "text": None, "error_kind": None, "usage": {}, "finish_reason": None}
        for index in range(attempt_offset, self.execution["network_retries"] + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise KeyboardInterrupt
            begin, tick, delay, retryable = time.time(), time.perf_counter(), 0.0, False
            result.update(status="error", text=None, http_status=None, error_kind=None)
            if on_attempt:
                on_attempt({"number": index + 1, "started_at": now(), "started_epoch": begin})
            request = urllib.request.Request(self.endpoint["base_url"].rstrip("/") + "/" + endpoint,
                                             data=body, headers=headers)
            attempt = {"number": index + 1, "started_at": now()}
            state = new_state() if stream else None
            capture = Capture(stream_directory, index + 1, self.key) if stream_directory is not None else None
            exception_class, reason_class = None, None
            try:
                with self.opener(request, timeout=self.execution["timeout_seconds"]) as response:
                    result["http_status"] = 200
                    value = read_stream(response, state, chat=messages is not None, capture=capture,
                                        cancel_event=cancel_event) if stream else json.load(response)
                choice = value["choices"][0]
                message = choice.get("message", {})
                text = message.get("content") if messages is not None else choice.get("text")
                refusal = message.get("refusal")
                result.update(http_status=200, usage=value.get("usage") or {}, provider_model=value.get("model"),
                              provider_response_id=value.get("id"), finish_reason=choice.get("finish_reason"))
                if text is None and isinstance(refusal, str):
                    text = refusal
                if choice.get("finish_reason") == "content_filter":
                    result.update(status="error", error_kind="provider_guardrail", text=text,
                                  error="Provider content filter", finish_reason="content_filter")
                elif not isinstance(text, str):
                    result.update(status="error", error_kind="api_protocol_error", error="API returned no text content")
                else:
                    result.update(status="ok", error_kind=None, text=text, error=None, http_status=200,
                                  finish_reason=choice.get("finish_reason"), usage=value.get("usage") or {},
                                  structured_refusal=refusal, provider_model=value.get("model"), provider_response_id=value.get("id"))
            except (urllib.error.HTTPError, StreamRemoteError) as error:
                exception_class = type(error).__name__
                detail = error.read().decode("utf-8", errors="replace") if isinstance(error, urllib.error.HTTPError) else str(error)
                code = error.code if type(error.code) is int else 400
                kind, retryable = classify_http(code, detail)
                delay = retry_after(error.headers.get("Retry-After") if isinstance(error, urllib.error.HTTPError) and error.headers else None)
                result.update(status="error", text=None, error_kind=kind, http_status=error.code,
                              error=detail.replace(self.key, "[REDACTED]")[:2000] if self.key else detail[:2000])
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
                    http.client.IncompleteRead, http.client.RemoteDisconnected, OSError) as error:
                retryable = True
                exception_class = type(error).__name__
                reason_class = type(error.reason).__name__ if isinstance(error, urllib.error.URLError) else None
                detail = f"{type(error).__name__}: {error}"
                result.update(status="error", text=None, error_kind="network_error",
                              error=detail.replace(self.key, "[REDACTED]")[:1000] if self.key else detail[:1000])
            except (ValueError, KeyError, IndexError, TypeError) as error:
                exception_class = type(error).__name__
                result.update(status="error", text=None, error_kind="api_protocol_error", error=f"Invalid API envelope: {type(error).__name__}")
            except BaseException:
                result.update(status="interrupted", error_kind="interrupted")
                raise
            finally:
                if capture:
                    captures.append(capture.finish(state, result))
            attempt.update(elapsed_seconds=time.perf_counter() - tick, error_kind=result["error_kind"], status=result["status"],
                http_status=result.get("http_status"), exception_class=exception_class, reason_class=reason_class,
                error=result.get("error"), finished_at=now())
            if stream:
                attempt.update(first_event_seconds=state["first_event_seconds"], first_text_seconds=state["first_text_seconds"],
                    received_characters=state["characters"], partial_output=result["status"] != "ok" and bool(state["characters"]))
            attempts.append(attempt)
            if not retryable or index == self.execution["network_retries"]:
                break
            delay = max(delay, min(self.execution["backoff_max_seconds"], self.execution["backoff_initial_seconds"] * 2 ** index))
            attempts[-1]["backoff_seconds"] = delay
            total_backoff += delay
            self.sleeper(delay)
        if not attempts:
            result.update(error_kind="network_retry_exhausted", error="Six transport attempts already recorded before resume")
        result.update(attempts=attempts, started_epoch=started, finished_epoch=time.time(),
                      elapsed_seconds=time.time() - started, backoff_seconds=total_backoff,
                      request_sha256=digest(payload), response_sha256=digest(result["text"]) if isinstance(result["text"], str) else None,
                      fatal=result["error_kind"] == "authentication_or_configuration_error" or
                            (result["error_kind"] == "http_error" and result.get("http_status") == 400))
        if stream:
            result.update(stream=True, stream_captures=captures,
                earlier_partial_attempts=sum(a.get("partial_output", False) for a in attempts[:-1]))
        return result
