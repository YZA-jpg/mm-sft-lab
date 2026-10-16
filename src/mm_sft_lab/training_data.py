"""Strict LoRA recipes and last-assistant supervision, importable without torch."""

import math
from pathlib import Path

from .common import digest, file_hash, now, read_json, read_jsonl, write_json, write_jsonl
from .step1_sources import inside

TRAIN_DEFAULTS = {
    "schema_version": 1, "protocol_id": "step1_lora_v1", "dataset": "textvqa",
    "model_path": "/data1/sharehub/models/Qwen3-VL-4B-Instruct", "epochs": 1,
    "dtype": "bfloat16", "attention": "sdpa", "seed": 42,
    "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "lora_bias": "none",
    "freeze_visual": True, "freeze_projector": True,
    "learning_rate": 5e-5, "optimizer": "adamw", "weight_decay": 0.0,
    "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-8, "max_grad_norm": 1.0,
    "scheduler": "linear", "warmup_ratio": 0.03,
    "micro_batch_size": 1, "gradient_accumulation_steps": 32, "effective_batch_size": 32,
    "max_tokens": 4096, "overlength_policy": "error", "gradient_checkpointing": True,
    "image_min_pixels": 65536, "image_max_pixels": 1048576,
    "save_steps": 100, "require_locked_data": True,
}


def load_train_config(path):
    value = read_json(path)
    if not isinstance(value, dict) or set(value) - set(TRAIN_DEFAULTS):
        raise ValueError("Unknown training configuration keys")
    result = {**TRAIN_DEFAULTS, **value}
    if result["schema_version"] != 1 or result["dataset"] not in ("textvqa", "hh", "vlguard"):
        raise ValueError("Select TextVQA, HH or VLGuard")
    if result["epochs"] != 1 or type(result["epochs"]) is not int:
        raise ValueError("Step 1 uses exactly one epoch")
    for name in ("lora_rank", "lora_alpha", "micro_batch_size", "gradient_accumulation_steps", "effective_batch_size",
                 "max_tokens", "save_steps", "image_min_pixels", "image_max_pixels"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if result["effective_batch_size"] != result["micro_batch_size"] * result["gradient_accumulation_steps"]:
        raise ValueError("Single-GPU effective batch must equal micro batch times accumulation")
    if result["dtype"] != "bfloat16" or result["attention"] != "sdpa" or result["optimizer"] != "adamw":
        raise ValueError("First training backend supports BF16/SDPA/AdamW")
    if result["scheduler"] not in ("linear", "cosine") or result["overlength_policy"] != "error":
        raise ValueError("Explicit linear/cosine scheduler and error-on-overlength are supported")
    for name in ("lora_dropout", "warmup_ratio", "weight_decay"):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]) or not 0 <= result[name] < 1:
            raise ValueError(f"Invalid {name}")
    for name in ("learning_rate", "adam_epsilon", "max_grad_norm"):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if (not isinstance(result["adam_betas"], list) or len(result["adam_betas"]) != 2 or
            any(type(v) not in (int, float) or not 0 <= v < 1 for v in result["adam_betas"])):
        raise ValueError("Adam betas must be two numbers in [0, 1)")
    if result["target_modules"] != ["q_proj", "k_proj", "v_proj", "o_proj"] or result["lora_bias"] != "none":
        raise ValueError("Recipe targets only language attention q/k/v/o LoRA, bias=none")
    for name in ("freeze_visual", "freeze_projector", "gradient_checkpointing", "require_locked_data"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if not result["freeze_visual"] or not result["freeze_projector"]:
        raise ValueError("Step 1 must freeze the complete visual branch and projector")
    if result["image_min_pixels"] > result["image_max_pixels"]:
        raise ValueError("Image pixel limits are reversed")
    for name in ("model_path", "protocol_id"):
        if not isinstance(result[name], str) or not result[name].strip():
            raise ValueError(f"{name} is required")
    return result


def validate_training_record(row, dataset):
    if row.get("dataset") != dataset or row.get("supervision") != "last_assistant" or not isinstance(row.get("id"), str):
        raise ValueError("Invalid training record identity/supervision")
    media = row.get("media_paths")
    if not isinstance(media, list) or len(media) != (1 if dataset in ("textvqa", "vlguard") else 0):
        raise ValueError("TextVQA/VLGuard need one image; HH must be text-only")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        raise ValueError("Training conversation must contain complete user/assistant pairs")
    for index, turn in enumerate(messages):
        if (set(turn) != {"role", "content"} or turn["role"] != ("user" if index % 2 == 0 else "assistant") or
                not isinstance(turn["content"], str) or not turn["content"].strip()):
            raise ValueError("Invalid training conversation")


def load_training_records(root, config):
    root = Path(root).resolve()
    manifest = read_json(root / "prepared/manifest.json")
    if manifest.get("step1_schema_version") != 1 or manifest.get("status") != "validated" or manifest.get("audit", {}).get("status") != "complete":
        raise ValueError("Run prepare_step1 before training")
    if config["require_locked_data"] and (manifest.get("provenance_status") != "locked" or manifest.get("is_mock")):
        raise ValueError("Real training requires locked data, not fixtures")
    filename = manifest.get("training_files", {}).get(config["dataset"])
    if not filename:
        raise ValueError(f"Dataset {config['dataset']} is absent from this training root; check --data-root")
    for name in (filename, "decontamination.json"):
        if file_hash(inside(root / "prepared", name)) != manifest["output_sha256"][name]:
            raise ValueError(f"Training data hash mismatch: {name}")
    records = read_jsonl(inside(root / "prepared", filename))
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Training data is empty or has duplicate IDs")
    images = {item["path"]: item["sha256"] for item in manifest["images"]}
    checked = set()
    for row in records:
        validate_training_record(row, config["dataset"])
        for name in row["media_paths"]:
            if name not in checked:
                if name not in images or file_hash(inside(root, name)) != images[name]:
                    raise ValueError(f"Training image changed: {name}")
                checked.add(name)
    return records, manifest


def training_messages(row, root):
    messages = []
    for index, turn in enumerate(row["messages"]):
        content = []
        if index == 0:
            content.extend({"type": "image", "image": str(inside(root, path))} for path in row["media_paths"])
        content.append({"type": "text", "text": turn["content"]})
        messages.append({"role": turn["role"], "content": content})
    return messages


def _ids(value):
    value = value.tolist() if hasattr(value, "tolist") else value
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Encode one training example at a time")
        value = value[0]
    return value


def supervised_labels(full_ids, prompt_ids, maximum, *, image_token_ids=()):
    if len(full_ids) > maximum:
        raise ValueError(f"Sequence has {len(full_ids)} tokens, exceeds {maximum}; no silent truncation")
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Assistant prompt is not an exact prefix; inspect the chat template before training")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = [-100 if token in image_token_ids else label for token, label in zip(full_ids, labels)]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("No supervised assistant tokens remain")
    return labels


def encode_training_record(processor, row, root, config):
    messages = training_messages(row, root)
    full = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                         return_dict=True, return_tensors="pt", padding=False)
    prompt = processor.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", padding=False)
    token_ids = tuple(value for value in (getattr(processor, "image_token_id", None), getattr(processor, "vision_start_token_id", None),
                                         getattr(processor, "vision_end_token_id", None), getattr(processor, "video_token_id", None)) if value is not None)
    ids = _ids(full["input_ids"])
    labels = supervised_labels(ids, _ids(prompt["input_ids"]), config["max_tokens"], image_token_ids=token_ids)
    has_pixels = "pixel_values" in full
    if bool(row["media_paths"]) != has_pixels or row["media_paths"] and "image_grid_thw" not in full:
        raise ValueError("Training media did not match processor visual inputs")
    summary = {"id": row["id"], "input_tokens": len(ids), "supervised_tokens": sum(label != -100 for label in labels[1:]),
               "labels_sha256": digest(labels), "input_ids_sha256": digest(ids), "images": len(row["media_paths"])}
    return full, labels, summary


def load_processor(config):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True, trust_remote_code=False)
    processor.tokenizer.padding_side = "right"
    image = processor.image_processor
    if not getattr(image, "do_resize", False):
        raise ValueError("Training image processor must enable resize for the explicit pixel budget")
    # In Transformers 4.57.1, from_dict can set legacy min/max attributes after
    # construction without updating size. Slow and fast Qwen processors consult
    # different fields, so synchronize the image component before its first call.
    expected = {"shortest_edge": config["image_min_pixels"], "longest_edge": config["image_max_pixels"]}
    image.size = dict(expected)
    image.min_pixels = expected["shortest_edge"]
    image.max_pixels = expected["longest_edge"]
    if (image.size != expected or image.min_pixels != expected["shortest_edge"] or
            image.max_pixels != expected["longest_edge"]):
        raise ValueError(f"Processor did not apply the training pixel budget: expected={expected}, "
                         f"size={image.size}, min_pixels={image.min_pixels}, max_pixels={image.max_pixels}")
    return processor


def processor_check(config, *, processor=None):
    """Check actual image resizing with synthetic inputs; never load model weights."""
    from PIL import Image
    processor = processor or load_processor(config)
    image = processor.image_processor
    patch, merge = image.patch_size, image.merge_size
    if type(patch) is not int or type(merge) is not int or patch < 1 or merge < 1:
        raise ValueError("Invalid processor patch_size/merge_size")
    minimum, maximum = config["image_min_pixels"], config["image_max_pixels"]
    sides = (max(1, min(64, math.isqrt(minimum) // 2)), max(2048, 2 * math.isqrt(maximum)))
    probes = []
    for side in sides:
        with Image.new("RGB", (side, side), (128, 128, 128)) as probe:
            encoded = image(images=probe, return_tensors="pt")
        grids = encoded.get("image_grid_thw")
        grids = grids.tolist() if hasattr(grids, "tolist") else grids
        if (not isinstance(grids, list) or len(grids) != 1 or len(grids[0]) != 3 or
                any(type(value) is not int or value < 1 for value in grids[0])):
            raise ValueError("Processor probe did not return one valid image_grid_thw")
        temporal, height, width = grids[0]
        if temporal != 1 or height % merge or width % merge:
            raise ValueError(f"Processor probe has an unexpected still-image grid: {grids[0]}")
        pixels = height * width * patch * patch
        if not minimum <= pixels <= maximum:
            raise ValueError(f"Processor resize probe outside training pixel budget: "
                             f"input={side}x{side}, grid={grids[0]}, pixels={pixels}, budget={minimum}..{maximum}")
        probes.append({"input_size": [side, side], "image_grid_thw": grids[0],
                       "resized_size": [width * patch, height * patch], "resized_pixels": pixels,
                       "visual_tokens": temporal * height * width // (merge * merge)})
    return {"schema_version": 1, "status": "passed", "synthetic_images": True,
            "scope": "Image processor resizing only; no model weights, training, datasets or API calls",
            "image_processor": type(image).__name__, "size": dict(image.size),
            "min_pixels": image.min_pixels, "max_pixels": image.max_pixels,
            "patch_size": patch, "merge_size": merge, "probes": probes}


def tokenization_environment():
    from .preflight import environment
    value = environment()
    # A CPU processor audit need not expose the training GPU, but library versions must agree.
    value.pop("cuda_visible_devices", None)
    return value


def token_audit(root, config, output, *, limit=None, processor=None):
    from .preflight import code_identity, model_identity
    records, manifest = load_training_records(root, config)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Audit limit must be positive")
    selected = records[:limit] if limit is not None else records
    processor = processor or load_processor(config)
    entries, errors = [], []
    for index, row in enumerate(selected):
        try:
            _, _, info = encode_training_record(processor, row, root, config)
            entries.append(info)
        except ValueError as error:
            errors.append({"id": row["id"], "error": str(error)})
        if (index + 1) % 500 == 0:
            print(f"Token audit: {index + 1}/{len(selected)}", flush=True)
    result = {"schema_version": 1, "created_at": now(), "status": "passed" if not errors else "failed",
              "full_dataset": len(selected) == len(records), "is_mock": manifest.get("is_mock", False),
              "config_sha256": digest(config), "data_manifest_sha256": file_hash(Path(root) / "prepared/manifest.json"),
              "model_fingerprint": None if manifest.get("is_mock") else model_identity(config["model_path"])["fingerprint"],
              "code_sha256": code_identity()["package_sha256"], "tokenization_environment": tokenization_environment(),
              "expected_records": len(records), "checked_records": len(selected), "errors": len(errors),
              "input_tokens": sum(row["input_tokens"] for row in entries), "supervised_tokens": sum(row["supervised_tokens"] for row in entries),
              "max_input_tokens": max((row["input_tokens"] for row in entries), default=0),
              "processor": processor.to_json_string() if hasattr(processor, "to_json_string") else "test_fixture"}
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Audit already exists; choose a new audit directory")
    write_jsonl(output / "tokens.jsonl", entries)
    write_jsonl(output / "errors.jsonl", errors)
    result["tokens_sha256"] = file_hash(output / "tokens.jsonl")
    write_json(output / "manifest.json", result)
    return result


class SFTCollator:
    def __init__(self, processor, root, config, audit_map=None):
        self.processor, self.root, self.config, self.audit_map = processor, root, config, audit_map

    def __call__(self, records):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        encoded = [encode_training_record(self.processor, row, self.root, self.config) for row in records]
        if self.audit_map is not None:
            for _, _, info in encoded:
                if info != self.audit_map.get(info["id"]):
                    raise ValueError(f"Tokenization changed since audit: {info['id']}")
        ids = [item[0]["input_ids"].squeeze(0) for item in encoded]
        labels = [torch.tensor(item[1], dtype=torch.long) for item in encoded]
        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Tokenizer has no pad token")
        batch = {"input_ids": pad_sequence(ids, batch_first=True, padding_value=pad),
                 "attention_mask": pad_sequence([torch.ones_like(value) for value in ids], batch_first=True, padding_value=0),
                 "labels": pad_sequence(labels, batch_first=True, padding_value=-100)}
        for key in ("pixel_values", "image_grid_thw"):
            values = [item[0][key] for item in encoded if key in item[0]]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch, [item[2] for item in encoded]


def train_plan(root, config):
    records, manifest = load_training_records(root, config)
    return {"status": "planned", "dataset": config["dataset"], "is_mock": manifest.get("is_mock", False),
            "training_records": len(records), "epochs": config["epochs"], "effective_batch_size": config["effective_batch_size"],
            "optimizer_steps": math.ceil(len(records) / config["effective_batch_size"]),
            "supervised_tokens": "pending_full_token_audit", "budget_comparison": "full_epoch_not_budget_matched",
            "config": config, "data_manifest_sha256": file_hash(Path(root) / "prepared/manifest.json")}
