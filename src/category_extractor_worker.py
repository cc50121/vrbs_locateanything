from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .category_extractor import CategoryResult, _parse_json, normalize_lm_category
from .prompts import CATEGORY_SYSTEM_PROMPT


def extract_one(tokenizer, model, question: str) -> CategoryResult:
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
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            repetition_penalty=1.05,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = _parse_json(raw)
    category, attributes, prompt = normalize_lm_category(parsed, question=question)
    return CategoryResult(category=category, attributes=attributes, prompt=prompt, source="small_lm_isolated", raw_response=raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch category extraction worker for isolated small-LM dependencies")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    questions = payload["questions"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if args.device == "auto":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    rows = [extract_one(tokenizer, model, question).__dict__ for question in questions]
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Loaded isolated small LM and extracted {len(rows)} category prompts: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
