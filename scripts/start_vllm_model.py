"""Start a detached, locally bound vLLM server; inspect/stop only its recorded process group."""

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/data1/sharehub/models/Qwen3-VL-4B-Instruct"
DEFAULT_SERVER = REPO / "runs/vllm-gpu0-port18000"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise ValueError(f"Locked: {path}; inspect the recorded PID before removing a stale lock") from None
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "created_at": now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def process_identity(pid):
    """Linux process birth token + launch nonce prevent PID-reuse cleanup."""
    try:
        folder = Path(f"/proc/{pid}")
        fields = (folder / "stat").read_text().split(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        environment = (folder / "environ").read_bytes().split(b"\0")
        return {"start_ticks": fields[19], "uid": folder.stat().st_uid,
                "pgid": os.getpgid(pid), "environment": environment}
    except (OSError, ValueError, IndexError):
        return None


def owned_alive(receipt):
    identity = process_identity(receipt["pid"])
    nonce = ("MM_SFT_VLLM_LAUNCH_ID=" + receipt["launch_id"]).encode()
    return bool(identity and identity["start_ticks"] == receipt["start_ticks"]
                and identity["uid"] == receipt["uid"] and identity["pgid"] == receipt["pid"]
                and nonce in identity["environment"])


def stop_server(folder):
    folder = Path(folder)
    receipt = read_json(folder / "server.json")
    if not owned_alive(receipt):
        if process_identity(receipt["pid"]) is not None:
            raise ValueError("PID identity differs; refusing to stop this process")
        print("Recorded server is no longer running; no signal sent.")
        return
    os.killpg(receipt["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while owned_alive(receipt) and time.monotonic() < deadline:
        time.sleep(.25)
    if owned_alive(receipt):
        os.killpg(receipt["pid"], signal.SIGKILL)
    receipt.update(stopped_at=now(), status="stop_requested")
    write_json(folder / "server.json", receipt)
    print(f"Stopped owned server process group {receipt['pid']}.")


def checkpoint_identity(folder):
    folder = Path(folder).resolve()
    weights = sorted({*folder.glob("*.safetensors"), *folder.glob("pytorch_model*.bin")})
    if not (folder / "config.json").is_file() or not weights:
        raise ValueError("Expected a complete local HF model directory, not an adapter-only directory")
    return {"path": str(folder), "config_sha256": {p.name: sha_file(p) for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix in (".json", ".jinja")},
            "weight_files": [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights],
            "weight_identity_scope": "names/sizes/mtime, not full base-weight SHA256"}


def command(args):
    if not str(args.gpu).isdigit() or not 1 <= args.port <= 65535:
        raise ValueError("Use one physical GPU index and a valid port")
    if not math.isfinite(args.memory) or not 0 < args.memory < 1 or not math.isfinite(args.kv_cache_gib) or args.kv_cache_gib <= 0:
        raise ValueError("Invalid memory resource limits")
    # Preserve the venv executable path: resolving its symlink can select the base environment.
    python = os.path.abspath(os.path.expanduser(str(args.python)))
    return [python, "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(Path(args.model_path).resolve()), "--served-model-name", "target",
            "--host", "127.0.0.1", "--port", str(args.port), "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(args.memory),
            "--kv-cache-memory-bytes", str(int(args.kv_cache_gib * 1024 ** 3)),
            "--limit-mm-per-prompt", '{"image":1,"video":0}']


def runtime_versions(python):
    source = """import importlib.metadata as m, json, platform
versions = {}
for name in ('vllm', 'torch', 'torchvision', 'transformers', 'huggingface-hub'):
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
print(json.dumps({'python': platform.python_version(), 'packages': versions}))
"""
    env = dict(os.environ)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run([str(python), "-c", source], env=env, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("Cannot query vLLM environment: " + result.stderr[-1200:])
    value = json.loads(result.stdout)
    if not value["packages"]["vllm"]:
        raise ValueError("The selected Python environment does not contain vLLM")
    return value


def start(args):
    if not sys.platform.startswith("linux"):
        raise ValueError("Run model startup on the Linux server")
    folder = Path(args.server_dir).resolve()
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    args.model_path = Path(args.model_path).resolve()
    if not args.python.is_file():
        raise ValueError(f"Missing vLLM Python: {args.python}")
    argv = command(args)
    model = checkpoint_identity(args.model_path)
    settings = {"argv": argv, "gpu": str(args.gpu), "model": model, "launcher_sha256": sha_file(__file__),
                "runtime": runtime_versions(args.python)}
    signature = digest(settings)
    with file_lock(folder / ".start.lock"):
        path = folder / "server.json"
        if path.exists():
            previous = read_json(path)
            if owned_alive(previous):
                if previous["signature"] != signature:
                    raise ValueError("Owned server has different settings; stop it or use a different server directory/port")
                return previous
        try:
            with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                raise ValueError(f"Port {args.port} is occupied; no existing process was stopped")
        except ConnectionRefusedError:
            pass
        launch_id = uuid.uuid4().hex
        log = folder / ("server-" + launch_id[:12] + ".log")
        env = dict(os.environ)
        for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONNOUSERSITE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MM_SFT_VLLM_LAUNCH_ID=launch_id)
        with log.open("a", encoding="utf-8") as output:
            process = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(process.pid)
        if identity is None:
            raise RuntimeError(f"Server exited during startup; inspect {log}")
        receipt = {"schema_version": 1, "created_at": now(), "launch_id": launch_id, "pid": process.pid,
            "start_ticks": identity["start_ticks"], "uid": identity["uid"], "signature": signature,
            "settings": settings, "model_path": str(args.model_path), "model_id": "target", "gpu": str(args.gpu),
            "base_url": f"http://127.0.0.1:{args.port}/v1", "log": str(log), "status": "starting",
            "generation_parameters_sent": [],
            "memory_note": "Fixed KV cache overrides automatic cache sizing; weights/activations/graphs require additional VRAM."}
        write_json(path, receipt)
        write_json(folder / ("launch-" + launch_id[:12] + ".json"), receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--python", type=Path, default=REPO / ".venv-vllm/bin/python")
    parser.add_argument("--kv-cache-gib", type=float, default=40)
    parser.add_argument("--memory", "--gpu-memory-utilization", dest="memory", type=float, default=.65)
    parser.add_argument("--server-dir", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--stop", action="store_true")
    args = parser.parse_args(argv)
    args.server_dir = (args.server_dir or REPO / f"runs/vllm-gpu{args.gpu}-port{args.port}").resolve()
    if args.stop:
        stop_server(args.server_dir)
        return 0
    if args.status:
        receipt = read_json(args.server_dir / "server.json")
        print(json.dumps({"owned_process_alive": owned_alive(receipt), **receipt}, ensure_ascii=False, indent=2))
        return 0
    receipt = start(args)
    print(json.dumps({"status": "background_process_started_or_reused", "pid": receipt["pid"], "gpu": receipt["gpu"],
        "base_url": receipt["base_url"], "log": receipt["log"], "server_dir": str(args.server_dir),
        "next": "Run smoke_vllm_benchmarks.py; it waits for model readiness before generating."}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
