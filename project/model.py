"""
model.py -- Load SegFormer-B0 (ADE20K) for floor + window segmentation.

B0 is ~7x smaller than the previous B2 checkpoint and runs comfortably on a
worker thread at ~7-10 Hz on Apple Silicon. Same ADE20K label space so the
existing overlay/navigation code keeps working.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from project.mps_lock import MPS_LOCK


class _NullCtx:
    """No-op context manager for non-MPS devices."""
    def __enter__(self): return None
    def __exit__(self, *exc): return False

# SegFormer-B0 fine-tuned on ADE20K (150 indoor/outdoor classes including
# "floor", "wall", "windowpane", "door", which is what we care about).
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# Network input side. 384 keeps floor edges crisp without tanking throughput.
INPUT_SIZE = 384


def _pick_device() -> torch.device:
    """Choose the fastest available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str = MODEL_NAME):
    """
    Download (first run) and load the SegFormer processor + model.
    Returns (processor, model, device).
    """
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    device = _pick_device()
    model.to(device)
    model.eval()

    # Half precision on accelerator devices gives a healthy speed-up.
    if device.type in ("cuda", "mps"):
        model = model.to(dtype=torch.float16)

    return processor, model, device


def segment(image: Image.Image, processor, model, device) -> dict:
    """
    Run segmentation on a single PIL image.

    Returns a dict with:
        class_map  -- (H, W) int32 numpy array of predicted class IDs
        id2label   -- dict mapping class ID -> human-readable label
        latency_ms -- inference time in milliseconds
    """
    original_w, original_h = image.size

    # Resize on CPU to a fixed small input; cheaper than letting the processor
    # auto-pick (it defaults to ~512 which kills frame-rate on MPS).
    small = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    inputs = processor(images=small, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if model.dtype == torch.float16:
        pixel_values = pixel_values.to(dtype=torch.float16)

    start = time.perf_counter()
    # MPS backend is not thread-safe; serialise with the detector.
    need_lock = device.type == "mps"
    with (MPS_LOCK if need_lock else _NullCtx()):
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits

        # Up-sample logits back to the original frame size so the overlay aligns.
        logits = F.interpolate(
            logits.float(),
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        )

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Take the most likely class per pixel.
    class_map = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    return {
        "class_map": class_map,
        "id2label": model.config.id2label,
        "latency_ms": latency_ms,
    }
