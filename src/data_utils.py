from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2


@dataclass
class QueryItem:
    idx: int
    q_dir: str
    question: str
    human_image: str
    scene_image: str
    gt_label: str | None
    gt_boxes: list[list[int]]
    category: str = ""
    proposals: list[dict] = field(default_factory=list)
    tagged_image: str = ""
    locateanything_raw: str = ""


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_gt_annotations(base_dir: str | Path) -> list[dict]:
    gt_path = Path(base_dir) / "gts" / "items_annotations.json"
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("annotations", [])


def get_video_path(base_dir: str | Path) -> str:
    base_dir = Path(base_dir)
    candidates = [base_dir / "video" / "video.mp4", base_dir / "normal_video.mp4"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"No video found in {base_dir}")


def _query_sort_key(path: Path) -> int:
    m = re.search(r"q(\d+)$", path.name)
    return int(m.group(1)) if m else 10**9


def _find_one(q_dir: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(q_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file matching {patterns} in {q_dir}")


def _read_question(q_dir: Path, idx: int) -> str:
    candidates = [q_dir / "question.txt", q_dir / f"ask_{idx}.txt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"No question.txt or ask_{idx}.txt in {q_dir}")


def _normalize_relative(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def discover_queries(base_dir: str | Path, gt_annotations: list[dict]) -> list[QueryItem]:
    base_dir = Path(base_dir)
    gt_by_image: dict[str, list[dict]] = {}
    for ann in gt_annotations:
        image_path = _normalize_relative(ann.get("image_path", ""))
        gt_by_image.setdefault(image_path, []).append(ann)

    queries: list[QueryItem] = []
    for q_dir in sorted(base_dir.glob("q*"), key=_query_sort_key):
        if not q_dir.is_dir():
            continue
        idx = _query_sort_key(q_dir)
        human_image = _find_one(q_dir, [f"ask_human_{idx}.png", f"ask_human_{idx}.jpg", "ask_human_*.*"])
        scene_image = _find_one(q_dir, [f"ask_bg_{idx}.jpg", f"ask_bg_{idx}.png", "ask_bg_*.*"])
        question = _read_question(q_dir, idx)
        rel_scene = _normalize_relative(scene_image.relative_to(base_dir))
        anns = gt_by_image.get(rel_scene, [])
        gt_boxes = [[int(v) for v in ann["bbox"]] for ann in anns if "bbox" in ann]
        labels = [ann.get("label") for ann in anns if ann.get("label")]
        queries.append(
            QueryItem(
                idx=idx,
                q_dir=str(q_dir),
                question=question,
                human_image=str(human_image),
                scene_image=str(scene_image),
                gt_label=labels[0] if labels else None,
                gt_boxes=gt_boxes,
            )
        )
    return queries


def calculate_iou(box_a, box_b) -> float:
    if not box_a or not box_b:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-5)


def draw_eval_image(scene_image: str, gt_boxes: list[list[int]], pred_box, output_path: str | Path) -> None:
    img = cv2.imread(scene_image)
    if img is None:
        return
    for gt in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in gt]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 3)
        _draw_label(img, "GT", x1, y1, (0, 180, 0))
    if pred_box:
        x1, y1, x2, y2 = [int(v) for v in pred_box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 230), 3)
        _draw_label(img, "Pred", x1, y1, (0, 0, 230))
    ensure_dir(Path(output_path).parent)
    cv2.imwrite(str(output_path), img)


def _draw_label(img, text: str, x: int, y: int, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(img, (x, y0), (x + tw + 8, y0 + th + baseline + 6), color, -1)
    cv2.putText(img, text, (x + 4, y0 + th + 2), font, scale, (255, 255, 255), thickness)

