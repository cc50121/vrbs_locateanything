from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path

from PIL import Image


class LocateAnythingAdapter:
    def __init__(self, model_path: str, repo_path: str | None = None, device: str = "cuda", dtype: str = "bfloat16"):
        if repo_path and os.path.isdir(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            import torch
            from locateanything_worker import LocateAnythingWorker
        except Exception as exc:
            raise RuntimeError(
                "Could not import LocateAnythingWorker. Clone NVlabs/Eagle and set --locateanything-repo "
                "to Eagle/Embodied, or export PYTHONPATH to that directory."
            ) from exc

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch = torch
        self.worker_cls = LocateAnythingWorker
        _patch_locateanything_for_transformers5(model_path)
        self.worker = LocateAnythingWorker(model_path, device=device, dtype=dtype_map[dtype])
        self.model_path = model_path
        self.device = device

    def close(self) -> None:
        self.worker = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def propose(
        self,
        image_path: str,
        phrase: str,
        max_proposals: int = 12,
        generation_mode: str = "hybrid",
        temperature: float = 0.2,
    ) -> tuple[list[dict], str]:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        result = self.worker.ground_multi(
            img,
            phrase,
            generation_mode=generation_mode,
            max_new_tokens=1024,
            temperature=temperature,
            verbose=False,
        )
        answer = str(result.get("answer", ""))
        boxes = self._parse_boxes(answer, width, height)
        if not boxes:
            result = self.worker.detect(
                img,
                [phrase],
                generation_mode=generation_mode,
                max_new_tokens=1024,
                temperature=temperature,
                verbose=False,
            )
            answer = str(result.get("answer", ""))
            boxes = self._parse_boxes(answer, width, height)
        proposals = []
        for box in _dedupe_boxes(boxes)[:max_proposals]:
            proposals.append(
                {
                    "tag": len(proposals) + 1,
                    "bbox": [int(round(v)) for v in box],
                    "source": "locateanything",
                    "phrase": phrase,
                }
            )
        return proposals, answer

    def _parse_boxes(self, answer: str, image_width: int, image_height: int) -> list[list[float]]:
        parsed = []
        if hasattr(self.worker_cls, "parse_boxes"):
            try:
                for box in self.worker_cls.parse_boxes(answer, image_width, image_height):
                    parsed.append([box["x1"], box["y1"], box["x2"], box["y2"]])
            except Exception:
                parsed = []
        if not parsed:
            for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>|<(\d+)><(\d+)><(\d+)><(\d+)>", answer):
                groups = [g for g in match.groups() if g is not None]
                if len(groups) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in groups]
                parsed.append(
                    [
                        x1 / 1000 * image_width,
                        y1 / 1000 * image_height,
                        x2 / 1000 * image_width,
                        y2 / 1000 * image_height,
                    ]
                )
        return [_clamp_box(box, image_width, image_height) for box in parsed]


def _clamp_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted([max(0.0, min(float(width - 1), x1)), max(0.0, min(float(width - 1), x2))])
    y1, y2 = sorted([max(0.0, min(float(height - 1), y1)), max(0.0, min(float(height - 1), y2))])
    return [x1, y1, x2, y2]


def _dedupe_boxes(boxes: list[list[float]], iou_threshold: float = 0.95) -> list[list[float]]:
    kept: list[list[float]] = []
    for box in boxes:
        if (box[2] - box[0]) < 3 or (box[3] - box[1]) < 3:
            continue
        if all(_iou(box, prev) < iou_threshold for prev in kept):
            kept.append(box)
    return kept


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-5)


def _patch_locateanything_for_transformers5(model_path: str) -> None:
    """Make LocateAnything's HF remote code tolerate newer transformers kwargs.

    The official LocateAnything package pins transformers==4.57.1. The Qwen3-vl
    environment on this server currently has a newer transformers build that
    passes allow_all_kernels into _check_and_adjust_attn_implementation.
    """
    try:
        import inspect
        from pathlib import Path
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        patched_names = []
        for class_ref in [
            "modeling_locateanything.LocateAnythingPreTrainedModel",
            "modeling_qwen2.Qwen2PreTrainedModel",
        ]:
            base_cls = get_class_from_dynamic_module(class_ref, model_path, trust_remote_code=True)
            _patch_cached_source_signature(base_cls, Path(inspect.getsourcefile(base_cls)))
            if getattr(base_cls, "_vrbs_transformers5_patch", False):
                continue

            def patched_check(self, attn_implementation, is_init_check=False, _base_cls=base_cls, **kwargs):
                if attn_implementation == "magi":
                    return "magi"
                return super(_base_cls, self)._check_and_adjust_attn_implementation(
                    attn_implementation,
                    is_init_check=is_init_check,
                    **kwargs,
                )

            base_cls._check_and_adjust_attn_implementation = patched_check
            base_cls._vrbs_transformers5_patch = True
            patched_names.append(base_cls.__name__)
        model_cls = get_class_from_dynamic_module(
            "modeling_locateanything.LocateAnythingForConditionalGeneration",
            model_path,
            trust_remote_code=True,
        )
        _patch_locateanything_init_defaults(model_cls)
        if patched_names:
            print("Applied LocateAnything transformers>=5 compatibility patch: " + ", ".join(patched_names))
    except Exception as exc:
        print(f"Warning: could not apply LocateAnything compatibility patch: {exc}")


def _patch_cached_source_signature(base_cls, source_path: Path | None) -> None:
    if not source_path or not source_path.exists():
        return
    text = source_path.read_text(encoding="utf-8")
    old_sig = "def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):"
    new_sig = "def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False, **kwargs):"
    if old_sig not in text:
        updated = text
    else:
        updated = text.replace(old_sig, new_sig)
    updated = updated.replace(
        "return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check)",
        "return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check=is_init_check, **kwargs)",
    )
    updated = updated.replace(
        "self.rope_theta = config.rope_theta",
        "self.rope_theta = getattr(config, 'rope_theta', None) or getattr(config, 'rope_parameters', {}).get('rope_theta', 10000.0)",
    )
    marker = "self.language_model = Qwen2ForCausalLM(config.text_config)"
    if marker in updated and "_vrbs_rope_theta_compat" not in updated:
        updated = updated.replace(
            marker,
            (
                "if not hasattr(config.text_config, 'rope_theta'):\n"
                "                    config.text_config.rope_theta = getattr(config.text_config, 'rope_parameters', {}).get('rope_theta', 10000.0)\n"
                "                config.text_config._vrbs_rope_theta_compat = True\n"
                "                self.language_model = Qwen2ForCausalLM(config.text_config)"
            ),
        )
    if updated != text:
        source_path.write_text(updated, encoding="utf-8")
        print(f"Patched cached LocateAnything source for transformers>=5: {base_cls.__name__}")


def _patch_locateanything_init_defaults(model_cls) -> None:
    if getattr(model_cls, "_vrbs_init_defaults_patch", False):
        return
    original_init = model_cls.__init__

    def patched_init(self, config, *args, **kwargs):
        text_config = getattr(config, "text_config", None)
        if text_config is not None and not hasattr(text_config, "rope_theta"):
            rope_parameters = getattr(text_config, "rope_parameters", {}) or {}
            text_config.rope_theta = rope_parameters.get("rope_theta", 10000.0)
        return original_init(self, config, *args, **kwargs)

    model_cls.__init__ = patched_init
    model_cls._vrbs_init_defaults_patch = True
