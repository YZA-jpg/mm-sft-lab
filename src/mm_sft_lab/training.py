"""Single-GPU BF16 language LoRA, with token-normalized accumulation and resume."""

import hashlib
import math
import random
import time
from pathlib import Path

from .common import append_jsonl, digest, file_hash, now, read_json, read_jsonl, stage_lock, write_json
from .preflight import adapter_identity, code_identity, environment, inspect_local, model_identity
from .training_data import SFTCollator, load_processor, load_training_records, token_audit, tokenization_environment


def language_targets(names, suffixes):
    selected = [name for name in names if ".language_model.layers." in "." + name and
                name.rsplit(".", 1)[-1] in suffixes and ".visual." not in "." + name]
    if not selected or {name.rsplit(".", 1)[-1] for name in selected} != set(suffixes):
        raise ValueError("Could not locate all language attention LoRA target types")
    return sorted(selected)


def trainable_report(model):
    entries, total = [], 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            if (".language_model.layers." not in "." + name or ".visual." in "." + name or
                    not any(piece in name for piece in (".lora_A.", ".lora_B."))):
                raise ValueError(f"Unexpected trainable parameter: {name}")
            entries.append({"name": name, "parameters": parameter.numel(), "dtype": str(parameter.dtype)})
    if not entries:
        raise ValueError("No trainable language LoRA parameters")
    return {"trainable_parameters": sum(item["parameters"] for item in entries), "total_parameters": total,
            "trainable_tensors": entries, "visual_and_projector_frozen": True}


def lr_multiplier(completed, total, warmup, schedule):
    if completed < warmup:
        return completed / max(1, warmup)
    progress = min(1.0, max(0.0, (completed - warmup) / max(1, total - warmup)))
    return 1 - progress if schedule == "linear" else 0.5 * (1 + math.cos(math.pi * progress))


def parameter_hash(model, *, trainable):
    value = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            value.update(name.encode())
            value.update(str((tuple(parameter.shape), parameter.dtype)).encode())
            value.update(parameter.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return value.hexdigest()


def training_preflight(config):
    result = inspect_local(config["model_path"])
    import peft
    import torch
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("Training backend is single GPU only")
    result.update(peft_version=peft.__version__, config=config, status="environment_ready_not_training_verified")
    return result


def _load_model(config, resume_adapter=None):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model_path"], dtype=torch.bfloat16, attn_implementation=config["attention"],
        local_files_only=True, trust_remote_code=False)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    if resume_adapter:
        model = PeftModel.from_pretrained(base, str(resume_adapter), is_trainable=True, local_files_only=True)
    else:
        targets = language_targets([name for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)], config["target_modules"])
        model = get_peft_model(base, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                               lora_dropout=config["lora_dropout"], target_modules=targets,
                               bias=config["lora_bias"], task_type=TaskType.CAUSAL_LM))
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Accumulated adapter gradients stay FP32 even though the frozen base is BF16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.to("cuda:0")
    return model


def _audit_map(audit_dir, config, root, records, base_fingerprint, *, full):
    audit_dir = Path(audit_dir)
    audit = read_json(audit_dir / "manifest.json")
    if (audit.get("status") != "passed" or full and not audit.get("full_dataset") or audit.get("is_mock") or
            audit.get("config_sha256") != digest(config) or
            audit.get("data_manifest_sha256") != file_hash(Path(root) / "prepared/manifest.json") or
            audit.get("model_fingerprint") != base_fingerprint or
            audit.get("code_sha256") != code_identity()["package_sha256"] or
            audit.get("tokenization_environment") != tokenization_environment() or
            file_hash(audit_dir / "tokens.jsonl") != audit.get("tokens_sha256")):
        raise ValueError("Token audit is incomplete or differs from the current data/config/model")
    entries = read_jsonl(audit_dir / "tokens.jsonl")
    result = {row["id"]: row for row in entries}
    if len(result) != len(entries) or set(result) != {row["id"] for row in records}:
        raise ValueError("Token audit selection mismatch")
    return result


def _verify_checkpoint(path, signature):
    path = Path(path)
    metadata = read_json(path / "checkpoint.json")
    if metadata.get("run_fingerprint") != signature:
        raise ValueError("Checkpoint belongs to a different training recipe/data/environment")
    required = {"training_state.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"}
    if not required.issubset(metadata.get("files_sha256", {})):
        raise ValueError("Checkpoint is incomplete")
    for name, expected in metadata["files_sha256"].items():
        from .step1_sources import inside
        if file_hash(inside(path, name)) != expected:
            raise ValueError(f"Checkpoint file changed: {name}")
    return metadata


def _save_checkpoint(root, model, processor, optimizer, scheduler, progress, signature, torch):
    path = root / f"checkpoint-{progress['global_step']:06d}"
    if path.exists():
        raise ValueError("Checkpoint directory already exists; preserve it and resume from a completed earlier checkpoint")
    model.save_pretrained(path / "adapter", safe_serialization=True)
    processor.save_pretrained(path / "adapter")
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "progress": progress,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
             "python_rng": random.getstate()}
    torch.save(state, path / "training_state.pt")
    files = [item for item in path.rglob("*") if item.is_file()]
    write_json(path / "checkpoint.json", {"run_fingerprint": signature, "progress": progress,
               "files_sha256": {item.relative_to(path).as_posix(): file_hash(item) for item in files}})
    return path


def run_training(root, config, output, *, phase="train", audit_dir=None, limit=64, steps=2, resume_from=None):
    if phase not in ("smoke", "train"):
        raise ValueError("Select smoke or train")
    if phase == "smoke" and (type(limit) is not int or limit < 1 or type(steps) is not int or steps < 1):
        raise ValueError("Smoke limit and steps must be positive")
    records, data_manifest = load_training_records(root, config)
    if data_manifest.get("is_mock") or data_manifest.get("provenance_status") != "locked":
        raise ValueError("Synthetic fixtures cannot start real GPU training")
    if phase == "smoke":
        records = records[:limit]
    preflight = training_preflight(config)
    base = preflight["model"]
    output = Path(output).resolve()
    total_steps = math.ceil(len(records) / config["effective_batch_size"])
    if phase == "smoke":
        total_steps = min(total_steps, steps)
    if not total_steps:
        raise ValueError("No optimizer updates were selected")
    signature_data = {"config": config, "phase": phase, "selection_sha256": digest(records), "total_steps": total_steps,
                      "data_manifest_sha256": file_hash(Path(root) / "prepared/manifest.json"),
                      "base_fingerprint": base["fingerprint"], "code_sha256": code_identity()["package_sha256"],
                      "environment": environment()}
    signature = digest(signature_data)
    with stage_lock(output, "train"):
        path = output / "manifest.json"
        if resume_from:
            resume_from = Path(resume_from).resolve()
            if not resume_from.is_relative_to(output) or resume_from == output:
                raise ValueError("Resume checkpoint must be inside the selected training run")
            manifest = read_json(path)
            if manifest["run_fingerprint"] != signature or manifest["status"] == "complete":
                raise ValueError("Run differs or is already complete")
            _verify_checkpoint(resume_from, signature)
        else:
            if any(item.name != ".train.lock" for item in output.iterdir()):
                raise ValueError("New training run must use an empty directory")
            manifest = {"schema_version": 1, "created_at": now(), "status": "initialized", "is_mock": False,
                        "run_fingerprint": signature, "signature": signature_data, "selected_records": len(records),
                        "planned_optimizer_steps": total_steps}
            write_json(path, manifest)
            write_json(output / "resolved_config.json", config)
        try:
            import torch
            from transformers import set_seed
            set_seed(config["seed"])
            processor = load_processor(config)
            if phase == "smoke":
                audit_dir = Path(audit_dir) if audit_dir else output / "audit"
                if not (audit_dir / "manifest.json").exists():
                    token_audit(root, config, audit_dir, limit=len(records), processor=processor)
            elif audit_dir is None:
                raise ValueError("Formal training requires --audit-dir from a full token audit")
            token_map = _audit_map(audit_dir, config, root, records, base["fingerprint"], full=phase == "train")
            manifest["token_audit_sha256"] = file_hash(Path(audit_dir) / "manifest.json")
            model = _load_model(config, resume_from / "adapter" if resume_from else None)
            report = trainable_report(model)
            write_json(output / "trainable_parameters.json", report)
            frozen_before = parameter_hash(model, trainable=False)
            if manifest.get("frozen_parameters_sha256") not in (None, frozen_before):
                raise ValueError("Frozen base differs from the saved training run")
            manifest["frozen_parameters_sha256"] = frozen_before
            trained_before = parameter_hash(model, trainable=True)
            if not resume_from:
                manifest["initial_trainable_sha256"] = trained_before
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], betas=tuple(config["adam_betas"]),
                                         eps=config["adam_epsilon"], weight_decay=config["weight_decay"])
            warmup = math.floor(total_steps * config["warmup_ratio"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, total_steps, warmup, config["scheduler"]))
            progress = {"global_step": 0, "sample_offset": 0, "supervised_tokens": 0, "input_tokens": 0, "loss_sum": 0.0}
            if resume_from:
                state = torch.load(resume_from / "training_state.pt", map_location="cpu", weights_only=True)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                progress = state["progress"]
                for item in optimizer.state.values():
                    for key, value in item.items():
                        if isinstance(value, torch.Tensor) and key != "step":
                            item[key] = value.to("cuda:0")
                torch.set_rng_state(state["torch_rng"])
                torch.cuda.set_rng_state_all(state["cuda_rng"])
                random.setstate(state["python_rng"])
            order = list(range(len(records)))
            random.Random(config["seed"]).shuffle(order)
            collate = SFTCollator(processor, root, config, token_map)
            model.train()
            model.get_base_model().visual.eval()
            manifest.update(status="running", environment=environment(), warmup_steps=warmup)
            write_json(path, manifest)
            session = now()
            torch.cuda.reset_peak_memory_stats(0)
            while progress["sample_offset"] < len(order) and progress["global_step"] < total_steps:
                start = time.perf_counter()
                selected = order[progress["sample_offset"]:progress["sample_offset"] + config["effective_batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                token_count, input_count, loss_sum = 0, 0, 0.0
                for offset in range(0, len(selected), config["micro_batch_size"]):
                    batch, infos = collate([records[i] for i in selected[offset:offset + config["micro_batch_size"]]])
                    batch = {key: value.to("cuda:0") for key, value in batch.items()}
                    supervised = int((batch["labels"][:, 1:] != -100).sum().item())
                    if supervised != sum(info["supervised_tokens"] for info in infos) or not supervised:
                        raise ValueError("Supervision count differs from the audited causal labels")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss).item():
                        raise ValueError("Training loss is nonfinite")
                    (loss * supervised).backward()
                    loss_sum += float(loss.detach().float().item()) * supervised
                    token_count += supervised
                    input_count += sum(info["input_tokens"] for info in infos)
                    del batch, loss
                # Each micro loss is a token mean: normalize the accumulated token sum once.
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(token_count)
                norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
                if not float(norm.item()) > 0:
                    raise ValueError("All adapter gradients are zero")
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                progress = {"global_step": progress["global_step"] + 1, "sample_offset": progress["sample_offset"] + len(selected),
                            "supervised_tokens": progress["supervised_tokens"] + token_count,
                            "input_tokens": progress["input_tokens"] + input_count, "loss_sum": progress["loss_sum"] + loss_sum}
                torch.cuda.synchronize()
                append_jsonl(output / "updates.jsonl", {**progress, "session": session, "update_supervised_tokens": token_count,
                             "loss": loss_sum / token_count, "learning_rate": learning_rate, "gradient_norm": float(norm.item()),
                             "samples": len(selected), "wall_seconds": time.perf_counter() - start})
                print(f"[{progress['global_step']}/{total_steps}] loss={loss_sum / token_count:.6f} supervised={token_count}", flush=True)
                if progress["global_step"] % config["save_steps"] == 0 or progress["global_step"] == total_steps:
                    checkpoint = _save_checkpoint(output, model, processor, optimizer, scheduler, progress, signature, torch)
                    manifest["last_checkpoint"] = checkpoint.name
                    manifest["progress"] = progress
                    write_json(path, manifest)
            if parameter_hash(model, trainable=False) != frozen_before:
                raise ValueError("Frozen base/visual/projector parameters changed")
            if parameter_hash(model, trainable=True) == manifest["initial_trainable_sha256"]:
                raise ValueError("Adapter parameters did not change during training")
            adapter_dir = output / "adapter"
            model.save_pretrained(adapter_dir, safe_serialization=True)
            processor.save_pretrained(adapter_dir)
            write_json(adapter_dir / "training_provenance.json", {"base_fingerprint": base["fingerprint"],
                       "training_run_fingerprint": signature, "dataset": config["dataset"], "phase": phase})
            # Verify the serialized adapter tensor values, independently of the file inventory.
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
            expected = get_peft_model_state_dict(model)
            if saved.keys() != expected.keys() or any(not torch.equal(saved[name], tensor.detach().cpu()) for name, tensor in expected.items()):
                raise ValueError("Saved adapter tensors differ from trained adapter")
            manifest.update(status="complete", finished_at=now(), progress=progress,
                            frozen_parameters_unchanged=True, adapter_changed=True, saved_adapter_verified=True,
                            mean_token_loss=progress["loss_sum"] / progress["supervised_tokens"],
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0), adapter=adapter_identity(adapter_dir))
        except BaseException as error:
            manifest.update(status="failed", updated_at=now(), error=f"{type(error).__name__}: {str(error)[:1000]}")
            raise
        finally:
            write_json(path, manifest)
    return manifest


def merge_adapter(config, adapter_dir, output):
    """Optional standalone export; no training or inference is started."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Merged export requires an empty output directory")
    identity = adapter_identity(adapter_dir)
    provenance = read_json(Path(adapter_dir) / "training_provenance.json")
    base = model_identity(config["model_path"])
    if provenance["base_fingerprint"] != base["fingerprint"]:
        raise ValueError("Adapter was trained on a different base checkpoint")
    model = Qwen3VLForConditionalGeneration.from_pretrained(config["model_path"], dtype=torch.bfloat16,
                local_files_only=True, trust_remote_code=False, attn_implementation=config["attention"])
    model = PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True).merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    AutoProcessor.from_pretrained(config["model_path"], local_files_only=True).save_pretrained(output)
    write_json(output / "merge_provenance.json", {"base": base, "adapter": identity, "note": "BF16 merged export; validate inference before comparing"})
    return {"status": "exported", "output": str(output), "inference_validation": "pending"}
