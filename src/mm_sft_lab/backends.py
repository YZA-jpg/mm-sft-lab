"""Lazy imports keep offline validation independent of CUDA and API credentials."""

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from .benchmarks import messages_for
from .common import digest
from .preflight import environment, inspect_local


def api_identity(config):
    value = {}
    for key in ("base_url_env", "model_env"):
        resolved = os.environ.get(config[key], "").strip()
        if not resolved:
            raise ValueError(f"Set environment variable {config[key]}")
        value[key.removesuffix("_env")] = resolved
    parsed = urlparse(value["base_url"])
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API base URL must be an HTTP(S) service root without embedded credentials or query parameters")
    return {**value, "fingerprint": digest(value)}


def api_messages(messages):
    converted = []
    for message in messages:
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                parsed = urlparse(part["image"])
                if parsed.scheme != "file":
                    raise ValueError("Only local dataset images may be sent by this adapter")
                path = Path(url2pathname(parsed.path))
                mime = mimetypes.guess_type(path.name)[0]
                if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                    raise ValueError(f"Image format unsupported by Chat API: {path.suffix}")
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            else:
                raise ValueError("Unsupported message content")
        converted.append({"role": message["role"], "content": content})
    return converted


class ChatClient:
    def __init__(self, config):
        self.config = config
        self.identity = api_identity(config)
        self.key = os.environ.get(config["api_key_env"])
        if not self.key:
            raise ValueError(f"Set environment variable {config['api_key_env']} (local services may use a dummy key)")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.key, base_url=self.identity["base_url"],
                             timeout=config["timeout_seconds"], max_retries=config["max_retries"])

    def complete(self, messages, *, max_tokens, temperature, top_p=None, top_k=None, seed=None):
        kwargs = {"model": self.identity["model"], "messages": api_messages(messages),
                  self.config["token_parameter"]: max_tokens, "temperature": temperature}
        if top_p is not None:
            kwargs["top_p"] = top_p
        if seed is not None and self.config["send_seed"]:
            kwargs["seed"] = seed
        extra = dict(self.config["extra_body"])
        if top_k is not None:
            extra["top_k"] = top_k
        if extra:
            kwargs["extra_body"] = extra
        started = time.perf_counter()
        result = self.client.chat.completions.create(**kwargs)
        choice = result.choices[0]
        response = choice.message.content
        if response is None:
            response = getattr(choice.message, "refusal", None)
        if not isinstance(response, str):
            raise ValueError("API returned no text response")
        usage = result.usage
        return {
            "response": response, "finish_reason": choice.finish_reason,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "provider_response_id": result.id, "provider_model": result.model,
            "provider_fingerprint": getattr(result, "system_fingerprint", None),
            "seed_sent": seed if self.config["send_seed"] else None,
        }

    def safe_error(self, error):
        return f"{type(error).__name__}: {str(error).replace(self.key, '[REDACTED]')[:1000]}"


class MockBackend:
    def __init__(self, config):
        self.metadata = {"backend": "mock", "is_mock": True, "fingerprint": "mock_backend_v1"}

    def generate(self, records, data_root, config, seeds):
        return [{"response": next(iter(row["options"])) if row["task"] == "mcqa" else "MOCK RESPONSE: pipeline validation only.",
                 "finish_reason": "stop", "input_tokens": None, "output_tokens": None, "latency_ms": 0}
                for row in records]


class ApiBackend:
    def __init__(self, config):
        self.client = ChatClient(config["api"])
        self.metadata = {"backend": "chat_api", "is_mock": False, **self.client.identity, "environment": environment()}

    def generate(self, records, data_root, config, seeds):
        return [self.client.complete(messages_for(row, data_root), max_tokens=config["max_new_tokens"],
                                     temperature=config["temperature"] if config["do_sample"] else 0,
                                     top_p=config["top_p"], top_k=config["top_k"], seed=seed)
                for row, seed in zip(records, seeds)]


def trim_completion(tokens, eos_ids, pad_id, maximum):
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index + 1], "stop"
    if pad_id is not None:
        while tokens and tokens[-1] == pad_id:
            tokens = tokens[:-1]
    return tokens, "length" if len(tokens) >= maximum else "unknown"


class QwenBackend:
    def __init__(self, config):
        self.metadata = {"backend": "qwen3_vl_hf", "is_mock": False, **inspect_local(config["model_path"])}
        self.metadata["fingerprint"] = self.metadata["model"]["fingerprint"]
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True, trust_remote_code=False)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["model_path"], dtype=torch.bfloat16, attn_implementation=config["attention"],
            device_map={"": 0}, local_files_only=True, trust_remote_code=False,
        ).eval()
        self.metadata["processor"] = json.loads(self.processor.to_json_string())
        self.metadata["image_processor"] = json.loads(self.processor.image_processor.to_json_string())
        self.metadata["chat_template"] = self.processor.chat_template
        self.metadata["model_generation_config"] = self.model.generation_config.to_dict()
        self.metadata["attention"] = config["attention"]

    def generate(self, records, data_root, config, seeds):
        from transformers import set_seed
        set_seed(seeds[0])
        # Transformers 4.57 image loading expects filesystem paths, not file:// URIs.
        conversations = [messages_for(row, data_root, file_uris=False) for row in records]
        inputs = self.processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt", padding=True)
        if "pixel_values" not in inputs or inputs["pixel_values"].numel() == 0:
            raise ValueError("Processor did not produce visual inputs")
        if "image_grid_thw" not in inputs:
            raise ValueError("Processor did not produce image grids")
        grids = inputs["image_grid_thw"].tolist()
        if len(grids) != sum(len(row["media_paths"]) for row in records) or any(
            len(grid) != 3 or any(type(value) is not int or value <= 0 for value in grid) for grid in grids
        ):
            raise ValueError("Processor image grids do not match the requested images")
        self.torch.cuda.reset_peak_memory_stats(0)
        inputs = inputs.to("cuda:0")
        kwargs = {"max_new_tokens": config["max_new_tokens"], "do_sample": config["do_sample"],
                  "num_return_sequences": 1, "num_beams": 1, "return_dict_in_generate": False, "output_scores": False}
        if config["do_sample"]:
            kwargs.update(temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"])
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        self.torch.cuda.synchronize()
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        peak_allocated = self.torch.cuda.max_memory_allocated(0)
        peak_reserved = self.torch.cuda.max_memory_reserved(0)
        width = inputs["input_ids"].shape[1]
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        results = []
        grid_offset = 0
        for index, sequence in enumerate(generated):
            tokens, reason = trim_completion(sequence[width:].tolist(), eos_ids, self.processor.tokenizer.pad_token_id,
                                             config["max_new_tokens"])
            input_tokens = int(inputs["attention_mask"][index].sum().item())
            image_count = len(records[index]["media_paths"])
            results.append({
                "response": self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                "finish_reason": reason, "input_tokens": input_tokens,
                "output_tokens": len(tokens), "latency_ms": elapsed, "actual_batch_size": len(records),
                "padded_input_tokens": width, "pad_tokens": width - input_tokens,
                "image_count": image_count, "image_grid_thw": grids[grid_offset:grid_offset + image_count],
                "pixel_values_present": True, "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
            })
            grid_offset += image_count
        return results


def make_backend(config):
    return {"mock": MockBackend, "chat_api": ApiBackend, "qwen3_vl_hf": QwenBackend}[config["backend"]](config)
