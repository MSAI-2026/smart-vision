"""
model.py -- Load the SegFormer segmentation model and run inference on a single frame.
"""

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# Pre-trained SegFormer model fine-tuned on ADE20K (150 indoor/outdoor classes).
# b0 is ~4x faster than b2 with minor accuracy loss for our coarse categories.
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# Input size fed to the network. 512 matches the checkpoint; drop to 384 for more speed.
INFER_SIZE = 512

# ImageNet normalization (SegFormer's standard preprocessing).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def _pick_device() -> torch.device:
    """Choose the fastest available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str = MODEL_NAME):
    """
    Download (first run) and load the SegFormer model.
    Returns (processor, model, device).  The `processor` slot is None — we do
    our own preprocessing for speed, but the signature is kept for compatibility.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    device = _pick_device()
    model.to(device)
    model.eval()

    return None, model, device


def segment(frame_rgb: np.ndarray, processor, model, device) -> dict:
    """
    Run segmentation on a single RGB uint8 frame (H, W, 3).

    Returns a dict with:
        class_map  -- (H, W) int32 numpy array of predicted class IDs
        id2label   -- dict mapping class ID -> human-readable label
        latency_ms -- inference time in milliseconds
    """
    original_h, original_w = frame_rgb.shape[:2]

    # Resize + normalize directly with cv2/numpy (skip PIL + HF processor overhead).
    resized = cv2.resize(frame_rgb, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) - _MEAN) / _STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(pixel_values=tensor).logits
        # Argmax at low resolution (1 channel) before upsampling — much cheaper
        # than bilinear-upsampling all 150 logit channels.
        low_res_classes = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Nearest-neighbor upsample the class index map back to the original frame size.
    class_map = cv2.resize(
        low_res_classes,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    return {
        "class_map": class_map,
        "id2label": model.config.id2label,
        "latency_ms": latency_ms,
    }
