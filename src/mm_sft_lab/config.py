"""Explicit configurations; unknown keys are rejected instead of ignored."""

from __future__ import annotations

from .common import read_json

BASELINE_TASKS = ("siuo/gen", "siuo/mcqa", "mssbench/chat", "mssbench/embodied")
TASKS = BASELINE_TASKS + ("wildjailbreak/harmful", "wildjailbreak/benign", "textvqa/val", "textvqa/test")
API_DEFAULTS = {
    "base_url_env": "MODEL_BASE_URL", "api_key_env": "MODEL_API_KEY", "model_env": "MODEL_ID",
    "token_parameter": "max_tokens", "send_seed": False, "timeout_seconds": 120,
    "max_retries": 2, "extra_body": {},
}
DEFAULTS = {
    "schema_version": 1, "backend": "qwen3_vl_hf", "protocol_id": "controlled_v1",
    "model_path": "/data1/sharehub/models/Qwen3-VL-4B-Instruct",
    "dtype": "bfloat16", "attention": "sdpa", "batch_size": 1,
    "max_new_tokens": 1024, "do_sample": False, "seeds": [42],
    "temperature": None, "top_p": None, "top_k": None,
    "tasks": list(BASELINE_TASKS), "per_task_limit": None, "require_locked_data": True,
    "adapter_path": None,
    "api": API_DEFAULTS,
}


def positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def api_config(value, *, judge=False):
    defaults = {**API_DEFAULTS}
    if judge:
        defaults.update(base_url_env="JUDGE_BASE_URL", api_key_env="JUDGE_API_KEY", model_env="JUDGE_MODEL")
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError("Unknown API configuration keys; configure credentials through environment variable names")
    result = {**defaults, **value}
    for key in ("base_url_env", "api_key_env", "model_env"):
        if not isinstance(result[key], str) or not result[key].isidentifier():
            raise ValueError(f"{key} must be an environment variable name")
    if result["token_parameter"] not in ("max_tokens", "max_completion_tokens"):
        raise ValueError("Unsupported API token parameter")
    if type(result["send_seed"]) is not bool:
        raise ValueError("send_seed must be boolean")
    positive_integer(result["timeout_seconds"], "timeout_seconds")
    if type(result["max_retries"]) is not int or result["max_retries"] not in range(6):
        raise ValueError("max_retries must be between 0 and 5")
    if not isinstance(result["extra_body"], dict):
        raise ValueError("extra_body must be an object")
    if set(result["extra_body"]) & {"api_key", "messages", "model", "n", "stream", "seed", "max_tokens", "max_completion_tokens", "temperature", "top_p"}:
        raise ValueError("extra_body cannot override credentials or core generation parameters")
    return result


def load_config(path):
    value = read_json(path)
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown inference configuration keys")
    result = {**DEFAULTS, **value, "api": api_config(value.get("api", {}))}
    if result["schema_version"] != 1 or result["backend"] not in ("qwen3_vl_hf", "chat_api", "mock"):
        raise ValueError("Unsupported schema or backend")
    if result["adapter_path"] is not None and (result["backend"] != "qwen3_vl_hf" or not isinstance(result["adapter_path"], str) or not result["adapter_path"].strip()):
        raise ValueError("adapter_path requires a local HF backend and a nonempty path")
    if not isinstance(result["protocol_id"], str) or not result["protocol_id"].strip():
        raise ValueError("protocol_id is required")
    for key in ("batch_size", "max_new_tokens"):
        positive_integer(result[key], key)
    if result["per_task_limit"] is not None:
        positive_integer(result["per_task_limit"], "per_task_limit")
    if type(result["do_sample"]) is not bool or type(result["require_locked_data"]) is not bool:
        raise ValueError("do_sample and require_locked_data must be boolean")
    seeds = result["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct nonnegative integers")
    if not result["do_sample"] and len(seeds) != 1:
        raise ValueError("Multiple greedy copies are not independent responses")
    if len(seeds) % 2 == 0 or (len(seeds) > 1 and result["tasks"] != ["siuo/gen"]):
        raise ValueError("Multi-response evaluation requires an odd number of samples and tasks=['siuo/gen']")
    if result["do_sample"]:
        if result["batch_size"] != 1:
            raise ValueError("Sampling requires batch_size=1 for stable per-request resume seeds")
        if not isinstance(result["temperature"], (int, float)) or not 0 < result["temperature"] <= 10:
            raise ValueError("Sampling temperature must be positive")
        if not isinstance(result["top_p"], (int, float)) or not 0 < result["top_p"] <= 1:
            raise ValueError("Sampling top_p must be in (0, 1]")
        positive_integer(result["top_k"], "top_k")
    elif any(result[key] is not None for key in ("temperature", "top_p", "top_k")):
        raise ValueError("Leave sampling parameters null for greedy inference")
    if result["backend"] == "chat_api" and result["batch_size"] != 1:
        raise ValueError("Chat API backend currently requires batch_size=1")
    if result["dtype"] != "bfloat16" or result["attention"] not in ("sdpa", "flash_attention_2"):
        raise ValueError("Local backend supports BF16 with sdpa or flash_attention_2")
    if not isinstance(result["tasks"], list) or not result["tasks"] or len(set(result["tasks"])) != len(result["tasks"]) or set(result["tasks"]) - set(TASKS):
        raise ValueError("Invalid task selection")
    return result


def load_judge_config(path):
    value = read_json(path)
    allowed = {"schema_version", "judge_protocol_id", "api", "max_new_tokens", "temperature"}
    if not isinstance(value, dict) or set(value) - allowed or value.get("schema_version", 1) != 1:
        raise ValueError("Invalid judge configuration")
    result = {"schema_version": 1, "judge_protocol_id": "lab_rubric_v1", "max_new_tokens": 512, "temperature": 0, **value}
    if result["judge_protocol_id"] != "lab_rubric_v1":
        raise ValueError("Only the versioned lab_rubric_v1 is implemented")
    positive_integer(result["max_new_tokens"], "judge max_new_tokens")
    if result["temperature"] != 0:
        raise ValueError("Baseline judging requires temperature=0")
    result["api"] = api_config(value.get("api", {}), judge=True)
    return result
