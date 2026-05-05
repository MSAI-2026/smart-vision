import io
import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocr import LabelOcrConfig, run_label_ocr  # noqa: E402
from label_llm import DEFAULT_SYSTEM, ollama_chat  # noqa: E402

app = FastAPI()

SAVE_DIR = Path("received_images")
SAVE_DIR.mkdir(exist_ok=True)

# Reuse a single LabelOcrConfig (fast preset, downscale long side to 1920) so the
# PaddleOCR pipeline stays warm between glasses requests.
OCR_CONFIG = LabelOcrConfig()

# LLM (local Ollama) settings. All overridable via env vars before launching the server.
LLM_ENABLED = os.environ.get("HEYCYAN_LLM", "1") not in ("0", "false", "False", "")
LLM_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
LLM_TIMEOUT = float(os.environ.get("HEYCYAN_LLM_TIMEOUT", "120"))
LLM_INSTRUCTIONS = os.environ.get(
    "HEYCYAN_LLM_INSTRUCTIONS",
    "Describe this product and its most important features for someone who cannot see the label. "
    "Keep it concise (2-4 short sentences) and natural to read aloud.",
)


def _blocks_to_speech(blocks: list[dict]) -> str:
    """Flatten OCR blocks (top-to-bottom reading order) into one speakable string."""
    lines: list[str] = []
    for block in blocks:
        text = block.get("text", "").strip()
        if text:
            lines.extend(t.strip() for t in text.splitlines() if t.strip())
    return " ".join(lines)


def _describe_with_llm(ocr_payload: dict) -> tuple[str | None, str | None]:
    """Send the OCR JSON to local Ollama and return (description, error)."""
    if not LLM_ENABLED:
        return None, "llm-disabled"
    user_text = json.dumps(
        {"ocr": ocr_payload, "instructions": LLM_INSTRUCTIONS}, indent=2
    )
    try:
        reply = ollama_chat(
            LLM_URL,
            LLM_MODEL,
            DEFAULT_SYSTEM,
            user_text,
            LLM_TIMEOUT,
        )
    except Exception as e:  # connection refused, model missing, timeout, etc.
        return None, f"{type(e).__name__}: {e}"
    reply = reply.strip()
    if not reply:
        return None, "empty-llm-reply"
    return reply, None


async def _run_pipeline_on_upload(file: UploadFile) -> dict:
    data = await file.read()

    filename = file.filename or "frame.jpg"
    (SAVE_DIR / filename).write_bytes(data)

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from e

    try:
        ocr_result = run_label_ocr(image, OCR_CONFIG)
    except ValueError:
        return {
            "blocks": [],
            "text": "",
            "description": None,
            "llm_error": "no-text-detected",
            "speech": "I could not read any text on the label.",
        }

    blocks = ocr_result.get("blocks", [])
    full_text = "\n\n".join(b["text"] for b in blocks).strip()
    flat_text = _blocks_to_speech(blocks)

    description, llm_error = _describe_with_llm(ocr_result)

    if description:
        speech = description
    elif flat_text:
        speech = flat_text
    else:
        speech = "I could not read any text on the label."

    if llm_error:
        print(f"[llm] skipped/failed: {llm_error}", file=sys.stderr)

    return {
        "blocks": blocks,
        "text": full_text,
        "description": description,
        "llm_error": llm_error,
        "speech": speech,
    }


# The phone client posts photos here; this is the endpoint it has always used.
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    return await _run_pipeline_on_upload(file)


# Kept as an alias so newer callers / tests can also use /analyze.
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    return await _run_pipeline_on_upload(file)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
