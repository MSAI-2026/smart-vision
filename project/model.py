"""
model.py -- Load the SegFormer segmentation model and run inference on a single frame.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Pre-trained SegFormer model fine-tuned on ADE20K (150 indoor/outdoor classes).
MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


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

    # Pre-process: the processor resizes/normalises for the network automatically.
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass (no gradient needed).
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Up-sample logits back to the original frame size so the overlay aligns.
    logits = F.interpolate(
        logits,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )

    # Take the most likely class per pixel.
    class_map = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    return {
        "class_map": class_map,
        "id2label": model.config.id2label,
        "latency_ms": latency_ms,
    }
