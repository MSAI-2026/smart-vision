"""
Semantic segmentation for Smart Vision.
Uses mit-semseg (CSAIL) with a pretrained ADE20K model.
Default: MobileNetV2 encoder + C1 decoder (fastest for real-time camera use).
"""

import os
import urllib.request
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from mit_semseg.models import ModelBuilder, SegmentationModule
    HAS_SEMSEG = True
except ImportError:
    HAS_SEMSEG = False

# ADE20K 150-class stable colour palette (BGR).
_rng = np.random.default_rng(42)
ADE20K_PALETTE = _rng.integers(0, 255, size=(150, 3), dtype=np.uint8)

# ADE20K class names (index 0-149).
ADE20K_CLASSES = [
    "wall","building","sky","floor","tree","ceiling","road","bed","windowpane",
    "grass","cabinet","sidewalk","person","earth","door","table","mountain",
    "plant","curtain","chair","car","water","painting","sofa","shelf","house",
    "sea","mirror","rug","field","armchair","seat","fence","desk","rock","wardrobe",
    "lamp","bathtub","railing","cushion","base","box","column","signboard",
    "chest of drawers","counter","sand","sink","skyscraper","fireplace","refrigerator",
    "grandstand","path","stairs","runway","case","pool table","pillow","screen door",
    "stairway","river","bridge","bookcase","blind","coffee table","toilet","flower",
    "book","hill","bench","countertop","stove","palm","kitchen island","computer",
    "swivel chair","boat","bar","arcade machine","hovel","bus","towel","light",
    "truck","tower","chandelier","awning","streetlight","booth","television",
    "airplane","dirt track","apparel","pole","land","bannister","escalator",
    "ottoman","bottle","buffet","poster","stage","van","ship","fountain",
    "conveyer belt","canopy","washer","plaything","swimming pool","stool","barrel",
    "basket","waterfall","tent","bag","minibike","cradle","oven","ball","food",
    "step","tank","trade name","microwave","pot","animal","bicycle","lake",
    "dishwasher","screen","blanket","sculpture","hood","sconce","vase","traffic light",
    "tray","ashcan","fan","pier","crt screen","plate","monitor","bulletin board",
    "shower","radiator","glass","clock","flag",
]

# Pretrained weights hosted by CSAIL.
_BASE_URL = "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-mobilenetv2dilated-c1_deepsup"
_ENCODER_URL = f"{_BASE_URL}/encoder_epoch_20.pth"
_DECODER_URL = f"{_BASE_URL}/decoder_epoch_20.pth"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _download(url: str, dest: str):
    """Download file if not already present."""
    if not os.path.isfile(dest):
        print(f"Downloading {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest)


@dataclass
class SegmentationResult:
    """Output of one segmentation pass."""
    mask: np.ndarray            # H×W int32, values 0-149
    present_classes: List[int] = field(default_factory=list)


class Segmentor:
    """CSAIL mit-semseg wrapper for real-time ADE20K segmentation."""

    def __init__(
        self,
        encoder_weights: Optional[str] = None,
        decoder_weights: Optional[str] = None,
        device: str = "cpu",
        input_size: int = 384,
    ):
        """
        Args:
            encoder_weights: Path to encoder .pth. Auto-downloads if None.
            decoder_weights: Path to decoder .pth. Auto-downloads if None.
            device:          Torch device ('cpu', 'cuda:0', 'mps').
            input_size:      Shorter edge resize before inference (lower = faster).
        """
        self._model: Optional[nn.Module] = None
        self._device = device
        self._input_size = input_size
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        if not HAS_SEMSEG:
            return

        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(models_dir, exist_ok=True)

        enc_path = encoder_weights or os.path.join(models_dir, "ade20k_encoder.pth")
        dec_path = decoder_weights or os.path.join(models_dir, "ade20k_decoder.pth")

        _download(_ENCODER_URL, enc_path)
        _download(_DECODER_URL, dec_path)

        encoder = ModelBuilder.build_encoder(
            arch="mobilenetv2dilated",
            fc_dim=320,
            weights=enc_path,
        )
        decoder = ModelBuilder.build_decoder(
            arch="c1_deepsup",
            fc_dim=320,
            num_class=150,
            weights=dec_path,
            use_softmax=True,
        )
        self._model = SegmentationModule(encoder, decoder, crit=nn.NLLLoss())
        self._model.to(device)
        self._model.eval()

    @property
    def available(self) -> bool:
        return self._model is not None

    def segment(self, frame: np.ndarray) -> Optional[SegmentationResult]:
        """
        Run segmentation on a BGR frame.
        Returns SegmentationResult or None if model unavailable.
        """
        if not self.available or frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize so shorter edge = input_size, keep aspect ratio.
        h, w = rgb.shape[:2]
        scale = self._input_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        tensor = self._transform(rgb_resized).unsqueeze(0).to(self._device)

        with torch.no_grad():
            scores = self._model({"img_data": tensor}, segSize=(new_h, new_w))

        _, pred = torch.max(scores, dim=1)
        mask = pred.squeeze(0).cpu().numpy().astype(np.int32)
        present = sorted(np.unique(mask).tolist())
        return SegmentationResult(mask=mask, present_classes=present)

    def draw_segmentation(
        self,
        frame: np.ndarray,
        result: SegmentationResult,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend colour-coded segmentation overlay onto frame."""
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(
            result.mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        colour_mask = ADE20K_PALETTE[mask_resized]
        return cv2.addWeighted(frame, 1 - alpha, colour_mask, alpha, 0)

    @staticmethod
    def class_name(class_id: int) -> str:
        if 0 <= class_id < len(ADE20K_CLASSES):
            return ADE20K_CLASSES[class_id]
        return str(class_id)


if __name__ == "__main__":
    seg = Segmentor(device="cpu")
    if not seg.available:
        print("mit-semseg not installed — run: pip install mit-semseg")
        exit(1)

    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = seg.segment(frame)
            if result:
                frame = seg.draw_segmentation(frame, result)
            cv2.imshow("ADE20K Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
