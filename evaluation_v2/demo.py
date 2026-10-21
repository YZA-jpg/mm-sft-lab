"""Harmless synthetic sources and fake HTTP envelopes for all eight benchmark paths."""

import io
import json
from pathlib import Path

from PIL import Image

from .common import write_json
from .data import Writer
from .transport import Client


def make_demo(root):
    writer = Writer(root)
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(stream, format="PNG")
    image = writer.image(stream.getvalue(), root)
    rows = []

    def row(bench, split, identity, reference):
        rows.append({"id": f"{bench}/{split}/{identity}", "benchmark": bench, "split": split,
                     "query": f"Synthetic {bench} {split} question {identity}.",
                     "images": [] if bench == "wildjailbreak" else [image], "reference": reference})
    row("siuo", "gen", "one", {"safety_warning": "Synthetic safety warning.", "reference_answer": "Synthetic reference."})
    row("siuo", "gen", "two", {"safety_warning": "Synthetic safety warning.", "reference_answer": "Synthetic reference."})
    for split in ("chat", "embodied"):
        for condition in ("safe", "unsafe"):
            row("mssbench", split, condition, {"condition": condition, "pair_id": split + "pair"})
    row("figstep", "safebench", "one", {"scenario": "Synthetic scenario", "scenario_rule": "Do not fulfill harmful requests."})
    for kind in ("SSS", "SUU", "USU", "UUU", "SSU"):
        row("holisafe", kind, "one", {"type": kind})
    for split in ("harmful", "benign"):
        row("wildjailbreak", split, "one", {})
    for answer in ("A", "B"):
        row("mmstar", "test", answer.lower(), {"answer": answer, "choice_letters": list("ABCD")})
    row("mathvista", "testmini", "one", {"answer": "4", "question_type": "free_form", "answer_type": "integer", "precision": 0, "choices": None})
    row("mmvet", "v1", "one", {"answer": "Synthetic reference", "capability": ["recognition"]})
    for name, text in {"siuo": "DEMO SIUO: ", "mssbench": "DEMO MSS: ",
                       "holisafe": "DEMO HOLISAFE {question} {safety_label} {response}",
                       "mathvista": "DEMO MATHVISTA", "mmvet": "DEMO MMVET",
                       "wildguard": "DEMO WILDGUARD\nHuman user: {prompt}\nAI assistant: {response}"}.items():
        writer.prompt(name, text)
    return writer.finish(rows, is_mock=True, provenance={"kind": "synthetic_fixture_not_benchmark_data"})


class FakeHTTP:
    def __init__(self):
        self.calls = []

    def __call__(self, request, **kwargs):
        body = json.loads(request.data)
        self.calls.append({"url": request.full_url, "body": body})
        if set(body) not in ({"model", "messages"}, {"model", "prompt"}):
            raise AssertionError("Unexpected generation parameters")
        if body["model"] == "target":
            text = "4" if "Synthetic mathvista" in json.dumps(body) else "A"
        elif "prompt" in body:
            harmful = "Synthetic wildjailbreak harmful" in body["prompt"]
            text = f"Harmful request: {'yes' if harmful else 'no'}\nResponse refusal: {'no' if harmful else 'yes'}\nHarmful response: {'yes' if harmful else 'no'}"
        else:
            content = json.dumps(body["messages"], ensure_ascii=False)
            if "DEMO SIUO" in content:
                text = json.dumps({"safety": False, "effectiveness": "ineffective"})
            elif "DEMO MSS" in content:
                text = '{"class_index":2}'
            elif "DEMO HOLISAFE" in content:
                text = "<label>True</label> The model's behaviour follows the rule."
            elif "DEMO MATHVISTA" in content:
                text = '{"answer":"4"}'
            elif "DEMO MMVET" in content:
                text = "0.7"
            else:
                text = "Result: [[unsafe]]"
        choice = {"message": {"content": text}, "finish_reason": "stop"} if "messages" in body else {"text": text, "finish_reason": "stop"}
        return io.BytesIO(json.dumps({"id": "fake-response", "model": body["model"], "choices": [choice],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode())

    def factory(self, endpoint, execution):
        # Fake keys never enter an HTTP header or real network call.
        return Client({**endpoint, "api_key_env": None}, execution, opener=self, sleeper=lambda _: None)
