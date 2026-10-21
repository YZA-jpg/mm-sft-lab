"""Own only the vLLM processes started here; target and WildGuard use one GPU in phases."""

import importlib.metadata
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .common import append, file_hash, latest, lock, now, read_json, read_journal, write_json
from .metrics import export_report
from .runner import generation_jobs, initialize, load_run, stage


def checkpoint_identity(path, *, label=None, adapter=None):
    root = Path(path).resolve()
    if not (root / "config.json").is_file():
        raise ValueError("Expected a complete local HF checkpoint directory, not just a LoRA adapter")
    weights = sorted({*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")})
    if not weights:
        raise ValueError("Checkpoint has no local weights")
    configs = {p.name: file_hash(p) for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".jinja")}
    result = {"kind": "local_checkpoint", "path": str(root), "label": label or root.name,
              "config_sha256": configs,
              "weights": [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights],
              "weight_hash_scope": "base weight names/sizes/mtime; config bytes SHA256; adapter full bytes SHA256"}
    if adapter:
        folder = Path(adapter).resolve()
        if not (folder / "adapter_config.json").is_file() or not any(folder.glob("adapter_model.*")):
            raise ValueError("Invalid LoRA directory")
        result["adapter"] = {"path": str(folder), "files_sha256": {p.name: file_hash(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    return result


def runtime_info(python=sys.executable):
    source = "import importlib.metadata as m,json,platform; names=['vllm','torch','transformers']; print(json.dumps({'python':platform.python_version(),'packages':{n:m.version(n) if m.packages_distributions().get(n) else None for n in names}}))"
    result = subprocess.run([str(python), "-c", source], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("Cannot inspect vLLM Python: " + result.stderr[-1500:])
    return json.loads(result.stdout)


def server_command(python, model, name, port, *, adapter=None, memory=None, kv_cache_gib=None, image_only=False):
    args = [str(python), "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
            "--served-model-name", name, "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1"]
    if adapter:
        args += ["--enable-lora", "--lora-modules", "target_adapter=" + str(adapter)]
    if memory is not None:
        if not 0 < memory < 1:
            raise ValueError("GPU memory utilization must be between 0 and 1")
        args += ["--gpu-memory-utilization", str(memory)]
    if kv_cache_gib is not None:
        if type(kv_cache_gib) not in (int, float) or not math.isfinite(kv_cache_gib) or kv_cache_gib <= 0:
            raise ValueError("KV cache GiB must be positive and finite")
        args += ["--kv-cache-memory-bytes", str(int(kv_cache_gib * 1024 ** 3))]
    if image_only:
        args += ["--limit-mm-per-prompt", '{"image":1,"video":0}']
    return args


def local_models(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
        return json.load(response)


@contextmanager
def serve(python, model, name, port, gpu, output, *, adapter=None, memory=None, kv_cache_gib=None,
          image_only=False, startup_timeout=900):
    if os.name != "posix":
        raise ValueError("Real vLLM serving is a Linux server check; local preparation supports command planning")
    if not str(gpu).isdigit():
        raise ValueError("This protocol expects one physical GPU index")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            raise ValueError(f"Port {port} is occupied; do not stop another user's server")
    except (ConnectionRefusedError, socket.timeout):
        pass
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    args = server_command(python, model, name, port, adapter=adapter, memory=memory,
                          kv_cache_gib=kv_cache_gib, image_only=image_only)
    identity = checkpoint_identity(model, adapter=adapter)
    receipt = {"started_at": now(), "argv": args, "gpu": str(gpu), "identity": identity, "generation_parameters_sent": []}
    write_json(output / "launch.json", receipt)
    env = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.perf_counter()
    stream = (output / "server.log").open("a", encoding="utf-8")
    process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    write_json(output / "owner.json", {"pid": process.pid, "owned_process_group": process.pid, "created_at": now()})
    try:
        while time.perf_counter() - started < startup_timeout:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup; inspect {output / 'server.log'}")
            try:
                models = local_models(port)
                expected = "target_adapter" if adapter else name
                advertised = {m["id"]: m for m in models["data"]}
                if expected not in advertised:
                    raise ValueError("Expected model/adapter not advertised; refusing silent base-model fallback")
                if not adapter and str(Path(advertised[expected].get("root", "")).resolve()) != str(Path(model).resolve()):
                    raise ValueError("Advertised checkpoint root differs from requested model")
                if adapter and str(Path(advertised[expected].get("root", "")).resolve()) != str(Path(adapter).resolve()):
                    raise ValueError("Advertised LoRA root differs from requested adapter")
                receipt.update(ready_at=now(), startup_seconds=time.perf_counter() - started, advertised_models=models)
                write_json(output / "launch.json", receipt)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(1)
        else:
            raise TimeoutError("vLLM startup timeout; see server.log")
        yield receipt
    finally:
        if process.poll() is None:
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError("Owned process group identity changed; refusing cleanup")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        receipt.update(stopped_at=now(), total_service_seconds=time.perf_counter() - started, exit_code=process.returncode)
        write_json(output / "launch.json", receipt)
        stream.close()


class Monitor:
    def __init__(self, run_dir, gpu, interval=5):
        self.path = Path(run_dir) / "gpu-samples.jsonl"
        self.gpu, self.interval, self.stopped = str(gpu), interval, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stopped.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "-i", self.gpu,
                    "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.free,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                append(self.path, {"at": now(), "epoch": time.time(), "csv": result.stdout.strip(), "exit_code": result.returncode})
            except (OSError, subprocess.TimeoutExpired) as error:
                append(self.path, {"at": now(), "epoch": time.time(), "error": str(error)})
            self.stopped.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join(timeout=12)


def pending_count(run_dir, phase, *, retry_recoverable=False):
    manifest, records, _ = load_run(run_dir)
    config = manifest["signature"]["config"]
    saved = latest(read_journal(Path(run_dir) / phase / "results.jsonl"))
    generations = latest(read_journal(Path(run_dir) / "generation/results.jsonl"))
    needed = []
    for job in generation_jobs(records, config):
        if phase == "generation":
            needed.append(job["job_id"])
            continue
        selected = job["benchmark"] == "wildjailbreak" if phase == "wildguard" else job["benchmark"] not in ("wildjailbreak", "mmstar")
        if selected and generations.get(job["job_id"], {}).get("status") == "ok":
            needed += [job["job_id"] + f"#j{i}" for i in range(config["mmvet_judge_repeats"] if job["benchmark"] == "mmvet" else 1)]
    return sum(name not in saved or (retry_recoverable and saved[name].get("error_kind") in
               ("authentication_or_configuration_error", "local_preparation_error")) for name in needed)


def run_all(data_root, run_dir, config, model_path, wildguard_path, *, label=None, adapter=None, gpu="7",
            vllm_python=sys.executable, memory=None, target_kv_cache_gib=None, wildguard_kv_cache_gib=None,
            generate_only=False, resume=False):
    run_dir = Path(run_dir).resolve()
    identity = checkpoint_identity(model_path, label=label, adapter=adapter)
    if "wildjailbreak" in config["benchmarks"]:
        if not wildguard_path:
            raise ValueError("Provide the local WildGuard checkpoint before starting any GPU work")
        identity["wildguard_identity"] = checkpoint_identity(wildguard_path)
    identity["serving_runtime"] = {"python": str(vllm_python), "versions": runtime_info(vllm_python), "gpu": str(gpu),
        "memory_utilization": memory, "target_kv_cache_gib": target_kv_cache_gib,
        "wildguard_kv_cache_gib": wildguard_kv_cache_gib, "target_limit_mm_per_prompt": {"image": 1, "video": 0}}
    target_port = int(config["target"]["base_url"].split(":")[-1].split("/")[0])
    guard_port = int(config["wildguard"]["base_url"].split(":")[-1].split("/")[0])
    expected_target = "target_adapter" if adapter else "target"
    if config["target"]["model"] != expected_target or config["wildguard"]["model"] != "wildguard":
        raise ValueError("Managed serving expects target/target_adapter and wildguard model IDs")
    if config["target"]["base_url"] != f"http://127.0.0.1:{target_port}/v1" or config["wildguard"]["base_url"] != f"http://127.0.0.1:{guard_port}/v1":
        raise ValueError("Managed serving uses loopback endpoints; use individual phases to attach external servers")
    initialize(data_root, run_dir, config, identity, resume=resume)
    with lock(run_dir, "pipeline"), Monitor(run_dir, gpu):
        started = time.perf_counter()
        session = now()
        append(run_dir / "pipeline-sessions.jsonl", {"event": "started", "session_id": session, "at": now()})
        if pending_count(run_dir, "generation", retry_recoverable=resume):
            with serve(vllm_python, model_path, "target", target_port, gpu, run_dir / "services" / ("target-" + session.replace(":", "-")),
                       adapter=adapter, memory=memory, kv_cache_gib=target_kv_cache_gib, image_only=True):
                generated = stage(run_dir, "generation", retry_recoverable=resume)
                if generated["fatal"]:
                    raise ValueError("Generation stopped on configuration/authentication/preparation failure; fix it and resume before judging")
        if generate_only:
            append(run_dir / "pipeline-sessions.jsonl", {"event": "completed", "session_id": session,
                "at": now(), "wall_seconds": time.perf_counter() - started, "scope": "generation_only_no_judge_calls"})
            return export_report(run_dir)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            cloud = pool.submit(stage, run_dir, "deepseek", retry_recoverable=resume)
            if pending_count(run_dir, "wildguard", retry_recoverable=resume):
                if not wildguard_path:
                    raise ValueError("WildGuard model path required for pending WildJailbreak judgments")
                with serve(vllm_python, wildguard_path, "wildguard", guard_port, gpu,
                           run_dir / "services" / ("wildguard-" + session.replace(":", "-")), memory=memory,
                           kv_cache_gib=wildguard_kv_cache_gib):
                    stage(run_dir, "wildguard", retry_recoverable=resume)
            cloud.result()
        append(run_dir / "pipeline-sessions.jsonl", {"event": "completed", "session_id": session,
               "at": now(), "wall_seconds": time.perf_counter() - started})
        return export_report(run_dir)
