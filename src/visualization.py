from __future__ import annotations

from pathlib import Path

import cv2

from .data_utils import ensure_dir


COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (0, 102, 255),
    (102, 0, 204),
]


def render_tagged_image(image_path: str, proposals: list[dict], output_path: str | Path) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for proposal in proposals:
        tag = int(proposal["tag"])
        x1, y1, x2, y2 = [int(v) for v in proposal["bbox"]]
        color = COLORS[(tag - 1) % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        _draw_tag(img, str(tag), x1, y1, color)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), img)


def _draw_tag(img, text: str, x: int, y: int, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    box_w = max(30, tw + 2 * pad)
    box_h = max(30, th + baseline + 2 * pad)
    y0 = max(0, y - box_h)
    x0 = max(0, x)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), color, -1)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)
    cv2.putText(img, text, (x0 + pad, y0 + box_h - pad - baseline), font, scale, (255, 255, 255), thickness)

