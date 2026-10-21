"""User-run downloads: pin public revisions, isolate failures, and never execute fetched files."""

import fnmatch
import json
import stat
import urllib.request
import zipfile
from pathlib import Path

from .common import asset_path, file_hash, now, read_json, write_json

HF_SOURCES = {
    "holisafe": ("etri-vilab/holisafe-bench", "dataset", ["*.json", "images/**", "data/**", "README.md", "LICENSE*"]),
    "mmstar": ("Lin-Chen/MMStar", "dataset", ["*.parquet", "data/**", "README.md", "LICENSE*"]),
    "mathvista": ("AI4Math/MathVista", "dataset", ["*testmini*", "data/*testmini*", "README.md", "LICENSE*"]),
    "wildguard-model": ("allenai/wildguard", "model", ["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt", "README.md", "LICENSE*"]),
}
UPSTREAM = {
    "adashield": ("SaFo-Lab/AdaShield", ["system_prompts.py"]),
    "holisafe": ("youngwanLEE/holisafe", ["eval/eval_by_openai_parallel_v2.py", "LICENSE"]),
    "mathvista": ("lupantech/MathVista", ["evaluation/prompts/ext_ans.py", "evaluation/calculate_score.py", "LICENSE"]),
    "mmvet": ("yuweihao/MM-Vet", ["mm-vet_evaluator.ipynb", "LICENSE"]),
    "wildguard": ("allenai/wildguard", ["wildguard/utils.py", "wildguard/wildguard.py", "LICENSE.md"]),
}


def _request(url, output=None):
    request = urllib.request.Request(url, headers={"User-Agent": "mm-sft-lab-evaluation-v2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if output is None:
            return json.load(response)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        with temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(output)


def safe_extract(archive, output, *, patterns=None, strip_first=False):
    output = Path(output).resolve()
    seen = set()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            original = member.filename
            name = original.split("/", 1)[1] if strip_first and "/" in original else original
            if not name or member.is_dir():
                continue
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ZIP symlink rejected")
            path = asset_path(output, name)
            if name in seen:
                raise ValueError("Duplicate ZIP member")
            seen.add(name)
            if patterns and not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                continue
            payload = stream.read(member)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != payload:
                raise ValueError("Refusing to replace a different extracted source file")
            if not path.exists():
                path.write_bytes(payload)


def _lock(folder, source, resolver):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "source.lock.json"
    if path.exists():
        value = read_json(path)
        if value["source"] != source:
            raise ValueError("Source changed in existing download directory")
        return value
    value = {"schema_version": 2, "created_at": now(), "source": source, "revision": resolver()}
    write_json(path, value)
    return value


def _inventory(folder, lock):
    files = {p.relative_to(folder).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file() and ".cache" not in p.parts
             and p.name not in ("download_manifest.json", "source.lock.json") and not p.name.endswith(".partial")}
    write_json(folder / "download_manifest.json", {"lock": lock, "files": files})
    return {"status": "downloaded", "files": len(files), "revision": lock["revision"]}


def download_source(name, raw_root):
    root = Path(raw_root).resolve()
    if name in HF_SOURCES:
        from huggingface_hub import HfApi, snapshot_download
        repo, kind, patterns = HF_SOURCES[name]
        folder = root / name
        saved = _lock(folder, "hf:" + kind + ":" + repo, lambda: HfApi().repo_info(repo, repo_type=kind).sha)
        snapshot_download(repo_id=repo, repo_type=kind, revision=saved["revision"], local_dir=folder,
                          allow_patterns=patterns, max_workers=4)
        return _inventory(folder, saved)
    if name == "prompts":
        result = {}
        for key, (repo, paths) in UPSTREAM.items():
            try:
                folder = root / "upstream" / key
                saved = _lock(folder, "github:" + repo,
                              lambda repo=repo: _request(f"https://api.github.com/repos/{repo}/commits/main")["sha"])
                for path in paths:
                    destination = folder / path
                    if not destination.exists():
                        _request(f"https://raw.githubusercontent.com/{repo}/{saved['revision']}/{path}", destination)
                result[key] = _inventory(folder, saved)
            except Exception as error:
                result[key] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        return {"status": "downloaded" if all(v["status"] == "downloaded" for v in result.values()) else "incomplete", "sources": result}
    if name == "figstep":
        folder, repo = root / "figstep", "CryptoAILab/FigStep"
        saved = _lock(folder, "github:" + repo, lambda: _request(f"https://api.github.com/repos/{repo}/commits/main")["sha"])
        archive = folder / "source.zip"
        if not archive.exists():
            _request(f"https://codeload.github.com/{repo}/zip/{saved['revision']}", archive)
        safe_extract(archive, folder, strip_first=True, patterns=["data/question/safebench.csv", "data/images/SafeBench/*.png", "LICENSE", "README.md"])
        return _inventory(folder, saved)
    if name == "mmvet":
        folder = root / "mmvet"
        url = "https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip"
        saved = _lock(folder, url, lambda: "release-v1")
        archive = folder / "mm-vet.zip"
        if not archive.exists():
            _request(url, archive)
        safe_extract(archive, folder)
        return _inventory(folder, saved)
    raise ValueError("Unknown download source")


def download_all(names, root):
    result = {}
    for name in names:
        print(f"[download] {name}", flush=True)
        try:
            result[name] = download_source(name, root)
        except Exception as error:
            result[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps({name: result[name]}, ensure_ascii=False), flush=True)
    write_json(Path(root) / "download-summary.json", result)
    return result


def verify_sources(root):
    """Every manifested raw file is checked before native import."""
    root = Path(root)
    found = {}
    for path in root.rglob("download_manifest.json"):
        value = read_json(path)
        for name, item in value["files"].items():
            target = asset_path(path.parent, name)
            if not target.is_file() or file_hash(target) != item["sha256"]:
                raise ValueError(f"Raw source changed: {target}")
        found[path.parent.relative_to(root).as_posix()] = value["lock"]
    return found
