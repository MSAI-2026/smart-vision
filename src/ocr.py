# Package / retail label OCR: scene text + spatial grouping + reading order.
# Uses PaddleOCR (detection + recognition), then merges nearby boxes into visual
# blocks and orders text by lines (so e.g. a large "8" stays with "VITAMINS & NUTRIENTS").
# Defaults: --fast (mobile det for English, no line-orientation) and --max-side 1920.

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class TextSpan:
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def aabb(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def quad_to_aabb(quad: list[list[float]] | np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(quad, dtype=np.float64)
    xs, ys = arr[:, 0], arr[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def rect_min_distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2
    else:
        dx = 0.0
    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2
    else:
        dy = 0.0
    return float(np.hypot(dx, dy))


def _legacy_boxes_are_quad(box: object) -> bool:
    if box is None:
        return False
    try:
        arr = np.asarray(box, dtype=np.float64)
        return arr.size >= 8
    except Exception:
        return False


def parse_paddle_result_legacy(ocr_result: list | None) -> list[TextSpan]:
    """PaddleOCR 2.x style: [ [ [box, (text, conf)], ... ] ] per page."""
    if not ocr_result or ocr_result[0] is None:
        return []
    spans: list[TextSpan] = []
    for line in ocr_result[0]:
        if not line or len(line) < 2:
            continue
        box, rec = line[0], line[1]
        if rec is None or not _legacy_boxes_are_quad(box):
            continue
        text, conf = rec[0], float(rec[1])
        t = str(text).strip()
        if not t:
            continue
        x1, y1, x2, y2 = quad_to_aabb(box)
        spans.append(TextSpan(text=t, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2))
    return spans


def parse_paddle_predict_output(predict_result: list | None) -> list[TextSpan]:
    """
    PaddleOCR 3.x / PaddleX returns a list of OCRResult dict-like objects:
    rec_polys, rec_texts, rec_scores.
    """
    if not predict_result:
        return []
    first = predict_result[0]
    try:
        texts = first["rec_texts"]
        polys = first["rec_polys"]
    except (KeyError, TypeError, IndexError):
        return []

    scores_raw = first["rec_scores"] if "rec_scores" in first else None
    if scores_raw is None:
        scores = [1.0] * len(texts)
    else:
        scores = list(np.asarray(scores_raw).reshape(-1))

    spans: list[TextSpan] = []
    for i, (poly, text) in enumerate(zip(polys, texts)):
        t = str(text).strip()
        if not t:
            continue
        conf = float(scores[i]) if i < len(scores) else 1.0
        if not _legacy_boxes_are_quad(poly):
            continue
        x1, y1, x2, y2 = quad_to_aabb(poly)
        spans.append(TextSpan(text=t, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2))
    return spans


def cluster_blocks(spans: list[TextSpan], merge_gap_px: float) -> list[list[TextSpan]]:
    n = len(spans)
    if n == 0:
        return []
    uf = UnionFind(n)
    aabbs = [s.aabb for s in spans]
    for i in range(n):
        for j in range(i + 1, n):
            if rect_min_distance(aabbs[i], aabbs[j]) <= merge_gap_px:
                uf.union(i, j)
    groups: dict[int, list[TextSpan]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(spans[i])
    return list(groups.values())


def order_block_to_lines(block: list[TextSpan]) -> list[list[TextSpan]]:
    """
    Reading order: sort by (y1, x1), then group into lines using vertical overlap with
    the current row band so a tall digit beside smaller words stays on the same row.
    """
    if not block:
        return []
    by_read = sorted(block, key=lambda s: (s.y1, s.x1))
    lines: list[list[TextSpan]] = []
    current: list[TextSpan] = [by_read[0]]
    for s in by_read[1:]:
        ry1 = min(sp.y1 for sp in current)
        ry2 = max(sp.y2 for sp in current)
        overlap_h = max(0.0, min(ry2, s.y2) - max(ry1, s.y1))
        if overlap_h > 1.0:
            current.append(s)
        else:
            lines.append(sorted(current, key=lambda sp: sp.x1))
            current = [s]
    lines.append(sorted(current, key=lambda sp: sp.x1))
    return lines


def block_reading_string(lines: list[list[TextSpan]]) -> str:
    out: list[str] = []
    for line in lines:
        out.append(" ".join(s.text for s in line))
    return "\n".join(out)


def sort_blocks_top_left(blocks: list[list[TextSpan]]) -> list[list[TextSpan]]:
    def key(b: list[TextSpan]) -> tuple[float, float]:
        return (min(s.y1 for s in b), min(s.x1 for s in b))

    return sorted(blocks, key=key)


def run_paddle_ocr(
    ocr: Any,
    rgb: np.ndarray,
    *,
    predict_kwargs: dict[str, Any] | None = None,
) -> list[TextSpan]:
    # PaddleOCR 3.x+: predict(). Older wheels: ocr().
    kw = dict(predict_kwargs or {})
    if hasattr(ocr, "predict"):
        return parse_paddle_predict_output(ocr.predict(rgb, **kw))
    if hasattr(ocr, "ocr"):
        return parse_paddle_result_legacy(ocr.ocr(rgb, cls=True))
    return []


def maybe_downscale_rgb(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Returns (array, scale) where scale maps coords from resized image back to original."""
    if max_side <= 0:
        return rgb, 1.0
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb, 1.0
    scale = max_side / m
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, 1.0 / scale


def scale_spans(spans: list[TextSpan], s: float) -> list[TextSpan]:
    if s == 1.0 or not spans:
        return spans
    out: list[TextSpan] = []
    for sp in spans:
        out.append(
            TextSpan(
                text=sp.text,
                confidence=sp.confidence,
                x1=sp.x1 * s,
                y1=sp.y1 * s,
                x2=sp.x2 * s,
                y2=sp.y2 * s,
            )
        )
    return out


def capture_from_camera() -> Image.Image:
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture (preview appears when the camera is ready)")
    img_bgr = None
    while True:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and ret and frame is not None and frame.size > 0:
            img_bgr = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def default_merge_gap(w: int, h: int) -> float:
    return max(12.0, 0.032 * min(w, h))


@dataclass
class LabelOcrConfig:
    """Same defaults as the ocr.py CLI (fast + max-side 1920)."""

    lang: str = "en"
    fast: bool = True
    max_side: int = 1920
    merge_gap: float | None = None
    min_conf: float = 0.0
    det_limit_side_len: int | None = None
    no_textline_orientation: bool = False


def run_label_ocr(image: Image.Image, cfg: LabelOcrConfig | None = None) -> dict[str, Any]:
    """
    Run PaddleOCR + block grouping on a PIL RGB image.
    Returns ``{"merge_gap_px": float, "blocks": [...]}`` (blocks match ``--json`` output).
    """
    cfg = cfg or LabelOcrConfig()
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        print(
            "Missing dependency: install PaddleOCR and PaddlePaddle, e.g.\n"
            "  pip install paddleocr paddlepaddle\n"
            "On Apple Silicon, follow https://www.paddlepaddle.org.cn/install/quick",
            file=sys.stderr,
        )
        raise e

    w, h = image.size
    merge_gap = cfg.merge_gap if cfg.merge_gap is not None else default_merge_gap(w, h)

    use_textline_ori = not (cfg.fast or cfg.no_textline_orientation)

    ocr_kwargs: dict[str, Any] = dict(
        lang=cfg.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=use_textline_ori,
    )
    if cfg.fast and cfg.lang == "en":
        ocr_kwargs.pop("lang", None)
        ocr_kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
        ocr_kwargs["text_recognition_model_name"] = "en_PP-OCRv5_mobile_rec"
    elif cfg.fast and cfg.lang == "ch":
        ocr_kwargs["ocr_version"] = "PP-OCRv4"

    ocr = PaddleOCR(**ocr_kwargs)

    rgb = np.array(image, dtype=np.uint8)
    rgb_in, inv_scale = maybe_downscale_rgb(rgb, cfg.max_side)

    predict_kw: dict[str, Any] = {}
    if not use_textline_ori:
        predict_kw["use_textline_orientation"] = False
    if cfg.det_limit_side_len is not None:
        predict_kw["text_det_limit_side_len"] = cfg.det_limit_side_len
    elif cfg.fast:
        predict_kw["text_det_limit_side_len"] = 960

    spans = scale_spans(run_paddle_ocr(ocr, rgb_in, predict_kwargs=predict_kw), inv_scale)
    if cfg.min_conf > 0:
        spans = [s for s in spans if s.confidence >= cfg.min_conf]

    if not spans:
        raise ValueError("No text detected in image.")

    raw_blocks = cluster_blocks(spans, merge_gap)
    blocks = sort_blocks_top_left(raw_blocks)

    structured: list[dict[str, Any]] = []
    for bi, block in enumerate(blocks):
        lines = order_block_to_lines(block)
        line_strs = [" ".join(s.text for s in line) for line in lines]
        structured.append(
            {
                "block": bi,
                "text": "\n".join(line_strs),
                "lines": [
                    {
                        "text": " ".join(s.text for s in line),
                        "spans": [asdict(s) for s in line],
                    }
                    for line in lines
                ],
            }
        )

    return {"merge_gap_px": merge_gap, "blocks": structured}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label / packaging OCR: PaddleOCR + spatial blocks + line order."
    )
    p.add_argument("image", nargs="?", help="Path to an image (JPEG, PNG, …).")
    p.add_argument("--camera", action="store_true", help="Capture from the default webcam.")
    p.add_argument(
        "--lang",
        default="en",
        help="PaddleOCR language (default: en). See PaddleOCR docs for codes.",
    )
    p.add_argument(
        "--merge-gap",
        type=float,
        default=None,
        help="Max gap (pixels) to merge two boxes into one block. "
        "Default: ~3.2%% of the shorter image side (min 12px).",
    )
    p.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Drop detections with confidence below this (0–1). Default: keep all.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON: blocks with lines and raw spans.",
    )
    p.add_argument(
        "--fast",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fast preset (default: on): mobile detector for English, no text-line orientation, "
        "det limit 960. Use --no-fast for server detector + orientation (slower, sometimes better).",
    )
    p.add_argument(
        "--no-textline-orientation",
        action="store_true",
        help="Disable text-line orientation (redundant when --fast, which is the default).",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=1920,
        metavar="PX",
        help="If longest image side exceeds PX, downscale before OCR (default: 1920). Use 0 to disable.",
    )
    p.add_argument(
        "--det-limit-side-len",
        type=int,
        default=None,
        metavar="PX",
        help="Forward to Paddle detection resize (smaller = faster, may miss tiny text).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.camera and args.image:
        print("Use either --camera or an image path, not both.", file=sys.stderr)
        sys.exit(2)
    if args.camera:
        image = capture_from_camera()
    elif args.image:
        path = os.path.expanduser(args.image)
        if not os.path.isfile(path):
            print(f"Not a file: {path}", file=sys.stderr)
            sys.exit(1)
        image = Image.open(path).convert("RGB")
    else:
        print(
            "Usage: python ocr.py <image>   or   python ocr.py --camera\n"
            "  Example: python ocr.py ./iv.jpeg\n"
            "  Default: fast models + --max-side 1920. Slower full quality: --no-fast --max-side 0",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = LabelOcrConfig(
        lang=args.lang,
        fast=args.fast,
        max_side=args.max_side,
        merge_gap=args.merge_gap,
        min_conf=args.min_conf,
        det_limit_side_len=args.det_limit_side_len,
        no_textline_orientation=args.no_textline_orientation,
    )
    try:
        result = run_label_ocr(image, cfg)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    merge_gap = result["merge_gap_px"]
    structured = result["blocks"]

    if args.json:
        print(json.dumps({"merge_gap_px": merge_gap, "blocks": structured}, indent=2))
    else:
        print(f"merge_gap_px={merge_gap:.1f}  (tune with --merge-gap if grouping is off)\n")
        for item in structured:
            print(f"--- Block {item['block']} ---")
            print(item["text"])
            print()
        print("=== Full text (blocks top-to-bottom, line breaks inside blocks) ===")
        print("\n\n".join(b["text"] for b in structured))


if __name__ == "__main__":
    main()
