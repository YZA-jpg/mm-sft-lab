"""Only operational settings; generation hyperparameters are deliberately absent."""

import copy
from urllib.parse import urlsplit

from .common import read_json

BENCHMARKS = ("siuo", "mssbench", "figstep", "holisafe", "wildjailbreak", "mmstar", "mathvista", "mmvet")
DEFAULTS = {
    "schema_version": 2, "protocol_id": "evaluation_v2_draft_20261021", "protocol_status": "provisional",
    "benchmarks": list(BENCHMARKS), "repeats": {name: 3 if name == "siuo" else 5 if name == "figstep" else 1 for name in BENCHMARKS},
    "mmvet_judge_repeats": 1,
    "target": {"base_url": "http://127.0.0.1:8000/v1", "model": "target", "api_key_env": None},
    "deepseek": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "deepseek-v4.1-flash", "api_key_env": "DASHSCOPE_API_KEY"},
    "wildguard": {"base_url": "http://127.0.0.1:8001/v1", "model": "wildguard", "api_key_env": None},
    "execution": {"generation_concurrency": 4, "generation_stream": False, "deepseek_concurrency": 4, "wildguard_concurrency": 8,
                  "timeout_seconds": 180, "network_retries": 5, "backoff_initial_seconds": 1.0,
                  "backoff_max_seconds": 30.0, "per_benchmark_limit": None},
}


def validate(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown v2 keys; do not configure generation hyperparameters")
    result = copy.deepcopy(DEFAULTS)
    for key, item in value.items():
        if isinstance(result[key], dict):
            if not isinstance(item, dict) or set(item) - set(result[key]):
                raise ValueError(f"Unknown {key} settings")
            result[key].update(item)
        else:
            result[key] = item
    if result["schema_version"] != 2 or result["protocol_status"] not in ("provisional", "confirmed"):
        raise ValueError("Unsupported protocol/schema")
    if not isinstance(result["protocol_id"], str) or not result["protocol_id"].strip():
        raise ValueError("Protocol ID required")
    selected = result["benchmarks"]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected) - set(BENCHMARKS):
        raise ValueError("Invalid benchmark selection")
    for name, count in result["repeats"].items():
        if type(count) is not int or count < 1 or (name not in ("siuo", "figstep") and count != 1):
            raise ValueError("Only SIUO/FigStep support repeated target responses")
    if type(result["mmvet_judge_repeats"]) is not int or result["mmvet_judge_repeats"] < 1:
        raise ValueError("Invalid MMVet judge repeats")
    for role in ("target", "deepseek", "wildguard"):
        endpoint = result[role]
        url = urlsplit(endpoint["base_url"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Endpoint URL must not contain credentials or query parameters")
        if not isinstance(endpoint["model"], str) or not endpoint["model"].strip():
            raise ValueError("Endpoint model required")
        env = endpoint["api_key_env"]
        if env is not None and (not isinstance(env, str) or not env.isidentifier()):
            raise ValueError("Configure only the name of an API key environment variable")
    execution = result["execution"]
    if type(execution["generation_stream"]) is not bool:
        raise ValueError("generation_stream must be a boolean transport setting")
    for key in ("generation_concurrency", "deepseek_concurrency", "wildguard_concurrency", "timeout_seconds"):
        if type(execution[key]) is not int or execution[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if execution["network_retries"] != 5 or type(execution["network_retries"]) is not int:
        raise ValueError("Protocol requires five network retries (six total attempts)")
    for key in ("backoff_initial_seconds", "backoff_max_seconds"):
        if type(execution[key]) not in (int, float) or not 0 <= execution[key] <= 300:
            raise ValueError("Invalid backoff setting")
    limit = execution["per_benchmark_limit"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Invalid benchmark limit")
    return result


def load(path):
    return validate(read_json(path))
