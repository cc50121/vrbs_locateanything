from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .prompts import CATEGORY_SYSTEM_PROMPT


EXPLICIT_ATTRIBUTES = [
    "white",
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "silver",
    "gray",
    "grey",
    "pink",
    "purple",
    "brown",
    "orange",
    "gaming",
]

KNOWN_OBJECTS = [
    ("headphones", ["headphones", "headphone", "earphones", "earphone"]),
    ("game controller", ["game controller", "controller", "gamepad", "game pad"]),
    ("power cube", ["powercube", "power cube", "power strip", "adapter"]),
    ("laptop", ["gaming laptop", "laptop", "computer"]),
    ("phone", ["phone", "mobile phone", "cellphone"]),
    ("cup", ["cup", "mug"]),
    ("shoes", ["shoes", "shoe", "sneakers", "sneaker"]),
]


@dataclass
class CategoryResult:
    category: str
    attributes: list[str]
    prompt: str
    source: str
    raw_response: str = ""


class CategoryExtractor:
    def __init__(self, model_path: str | None = None, device: str = "auto", isolated_deps_path: str | None = None):
        self.model_path = model_path
        self.device = device
        self.isolated_deps_path = isolated_deps_path
        self.tokenizer = None
        self.model = None
        self._load_error = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            kwargs = {"trust_remote_code": True}
            if self.device == "auto":
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
            print(f"Loaded small LM for category extraction: {model_path}")
        except Exception as exc:
            print(f"Warning: failed to load small LM {model_path}: {exc}. Falling back to rules.")
            self._load_error = exc
            self.tokenizer = None
            self.model = None

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def extract(self, question: str) -> CategoryResult:
        results = self.extract_many([question])
        return results[0]

    def extract_many(self, questions: list[str]) -> list[CategoryResult]:
        if self.model is not None and self.tokenizer is not None:
            try:
                return [self._extract_with_lm(question) for question in questions]
            except Exception as exc:
                print(f"Warning: small LM extraction failed: {exc}. Falling back to isolated worker or rules.")
        isolated_results = self._extract_many_with_isolated_worker(questions)
        if isolated_results is not None:
            return isolated_results
        return [self._extract_with_rules(question) for question in questions]

    def _extract_many_with_isolated_worker(self, questions: list[str]) -> list[CategoryResult] | None:
        if not self.model_path or not self.isolated_deps_path:
            return None
        deps_path = Path(self.isolated_deps_path)
        if not deps_path.exists():
            return None

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="vrbs_category_") as tmpdir:
            input_path = Path(tmpdir) / "questions.json"
            output_path = Path(tmpdir) / "categories.json"
            input_path.write_text(json.dumps({"questions": questions}, ensure_ascii=False), encoding="utf-8")
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                [str(deps_path), str(project_root), existing_pythonpath] if existing_pythonpath else [str(deps_path), str(project_root)]
            )
            cmd = [
                sys.executable,
                "-m",
                "src.category_extractor_worker",
                "--model",
                self.model_path,
                "--device",
                self.device,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            try:
                completed = subprocess.run(cmd, cwd=project_root, env=env, text=True, capture_output=True, check=False)
            except Exception as exc:
                print(f"Warning: isolated small LM worker failed to start: {exc}. Falling back to rules.")
                return None
            if completed.stdout:
                print(completed.stdout.strip())
            if completed.returncode != 0:
                if completed.stderr:
                    print(completed.stderr.strip())
                print("Warning: isolated small LM worker failed. Falling back to rules.")
                return None
            rows = json.loads(output_path.read_text(encoding="utf-8"))
        return [CategoryResult(**row) for row in rows]

    def _extract_with_lm(self, question: str) -> CategoryResult:
        import torch

        messages = [
            {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Command: {question}\n"
                    'Return JSON only, e.g. {"category":"headphones","attributes":["white"],"prompt":"white headphones"}'
                ),
            },
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{CATEGORY_SYSTEM_PROMPT}\nCommand: {question}\nJSON:"
        inputs = self.tokenizer(text, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.05,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed = _parse_json(raw)
        category, attributes, prompt = normalize_lm_category(parsed, question=question)
        if not category or not prompt:
            fallback = self._extract_with_rules(question)
            fallback.source = "rules_after_lm_failure"
            fallback.raw_response = raw
            return fallback
        return CategoryResult(category=category, attributes=attributes, prompt=prompt, source="small_lm", raw_response=raw)

    def _extract_with_rules(self, question: str) -> CategoryResult:
        q = question.lower()
        attributes = _explicit_attributes(q)
        category = ""
        matched_phrase = ""
        for canonical, aliases in KNOWN_OBJECTS:
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", q):
                    category = canonical
                    matched_phrase = alias
                    break
            if category:
                break
        if not category:
            category = _last_nounish_span(q)
        prompt_parts = attributes + [matched_phrase or category]
        prompt = _clean_text(" ".join(dict.fromkeys([p for p in prompt_parts if p])))
        return CategoryResult(category=category, attributes=attributes, prompt=prompt, source="rules")


def _parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def normalize_lm_category(parsed: dict, question: str | None = None) -> tuple[str, list[str], str]:
    q = _clean_text(question or "")
    explicit_attrs = set(_explicit_attributes(q))
    raw_attributes = parsed.get("attributes", [])
    attr_values: list[str] = []
    object_type = ""
    if isinstance(raw_attributes, dict):
        object_type = _clean_text(raw_attributes.get("type") or raw_attributes.get("object") or "")
        for key, value in raw_attributes.items():
            key_clean = _clean_text(key)
            value_clean = _clean_text(value)
            if not value_clean or value_clean in {"unknown", "none", "n/a"}:
                continue
            if key_clean in {"type", "object", "category", "brand"}:
                continue
            attr_values.append(value_clean)
    elif isinstance(raw_attributes, list):
        attr_values = [_clean_text(a) for a in raw_attributes if _clean_text(str(a))]
    elif raw_attributes:
        attr_values = [_clean_text(raw_attributes)]

    category = _canonical_category(_clean_text(parsed.get("category") or object_type))
    if category in {"object", "item", "thing", "electronics", "device", "unknown"} and object_type:
        category = _canonical_category(object_type)

    attr_values = [attr for attr in attr_values if attr in explicit_attrs]

    prompt = _clean_text(parsed.get("prompt") or "")
    sentence_like = re.match(r"^(find|get|help|where|locate|search)\b", prompt or "") or len(prompt.split()) > 5
    prompt_attrs = [attr for attr in _explicit_attributes(prompt) if attr in explicit_attrs]
    if prompt and not sentence_like:
        prompt_category = _canonical_category(prompt)
        if prompt_category:
            category = prompt_category
    prompt = _clean_text(" ".join(dict.fromkeys([p for p in (prompt_attrs or attr_values) + [category] if p])))

    return category, attr_values, prompt


def _explicit_attributes(text: str) -> list[str]:
    clean = _clean_text(text)
    return [attr for attr in EXPLICIT_ATTRIBUTES if re.search(rf"\b{re.escape(attr)}\b", clean)]


def _canonical_category(text: str) -> str:
    clean = _clean_text(text)
    for canonical, aliases in KNOWN_OBJECTS:
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", clean):
                return canonical
    return clean


def _clean_text(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9 /_-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _last_nounish_span(text: str) -> str:
    text = re.sub(r"\b(where|help|find|get|can|did|put|placed|is|are|my|me|i|the|a|an|go)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = [w for w in text.split() if w]
    if not words:
        return "object"
    return " ".join(words[-3:])
