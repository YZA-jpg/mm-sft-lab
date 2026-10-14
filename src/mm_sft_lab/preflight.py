"""Local checkpoint identity and environment inspection without network loading."""

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from .common import digest, file_hash, read_json


def environment():
    packages = {}
    for name in ("torch", "transformers", "accelerate", "huggingface-hub", "Pillow", "openai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def model_identity(model_path):
    root = Path(model_path).expanduser().resolve()
    config = read_json(root / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"Expected a dense Qwen3-VL checkpoint, found {config.get('model_type')}")
    if not (root / "preprocessor_config.json").is_file() or not (root / "tokenizer_config.json").is_file():
        raise ValueError("Checkpoint processor or tokenizer configuration is missing")
    indices = list(root.glob("*.safetensors.index.json"))
    weights = list(root.glob("*.safetensors"))
    if not weights:
        raise ValueError("Checkpoint contains no safetensors weights")
    for index in indices:
        expected = set(read_json(index)["weight_map"].values())
        if any(not (root / name).is_file() for name in expected):
            raise ValueError(f"Checkpoint has missing shards: {index}")
    controls = {path.name: file_hash(path) for path in sorted(root.iterdir())
                if path.is_file() and path.suffix in (".json", ".jinja", ".txt")}
    inventory = [{"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                 for path in sorted(weights)]
    value = {"path": str(root), "controls_sha256": controls, "weights": inventory,
             "weight_fingerprint_mode": "size_mtime_inventory_not_full_weight_hash"}
    return {**value, "fingerprint": digest(value)}


def inspect_local(model_path):
    identity = model_identity(model_path)
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # noqa: F401
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; check the CUDA-enabled torch installation and visible GPU")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES=7 before running")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("The visible GPU does not support BF16")
    free, total = torch.cuda.mem_get_info(0)
    return {"model": identity, "environment": environment(), "gpu": {
        "logical_device": "cuda:0", "name": torch.cuda.get_device_name(0),
        "free_bytes": free, "total_bytes": total, "torch_cuda": torch.version.cuda,
    }}


def code_identity():
    package = Path(__file__).resolve().parent
    files = {path.name: file_hash(path) for path in sorted(package.glob("*.py"))}
    commit, dirty = None, None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=package, capture_output=True, text=True, timeout=5, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"package_sha256": digest(files), "files": files, "git_commit": commit, "git_dirty": dirty}
