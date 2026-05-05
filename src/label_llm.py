# Send label OCR JSON to a local LLM (Ollama) for cleanup / structuring / Q&A.
#
# Setup (one-time):
#   1. Install Ollama: https://ollama.com/download
#   2. Pull a model, e.g.  ollama pull qwen2.5:1.5b
#   3. pip install httpx
#
# Run (Ollama app or `ollama serve` must be running):
#   python label_llm.py iv.jpeg
#   python label_llm.py --camera
#   python label_llm.py --ocr-json ocr_out.json --instructions "List allergens if any."
#   OLLAMA_MODEL=llama3.2 python label_llm.py iv.jpeg   # override model

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

DEFAULT_SYSTEM = """You are an AI assistant for blind and low-vision users. Your job is to describe
what is on a product label from OCR text (the user message is JSON: blocks of text in rough reading order).

Always:
- Identify the product clearly (brand, type of item, variant or flavor if present).
- Explain the most important features someone would want to know at a glance: what it is for, key
  benefits or claims, size or count, warnings or allergens if they appear in the text, and how to
  use it if instructions appear.
- Use natural, conversational language suitable to read aloud or hear from a screen reader.
- Correct obvious OCR mistakes only when the intended words are clear from context; never invent
  ingredients, claims, or facts that are not supported by the text.
- If the OCR is too fragmented to be sure, say what you can infer and what is uncertain."""


def load_ocr_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "blocks" not in data:
        print("OCR JSON must contain a top-level 'blocks' key (same format as ocr.py --json).", file=sys.stderr)
        sys.exit(1)
    return data


def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout: float,
) -> str:
    import httpx

    url = f"{base_url}/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=body)
    except httpx.ConnectError as e:
        print(
            "Could not connect to Ollama. Is it running?\n"
            "  Start: ollama serve   (or launch the Ollama app)\n"
            "  Then:  ollama pull " + model,
            file=sys.stderr,
        )
        raise e
    if r.status_code != 200:
        print(r.text, file=sys.stderr)
        r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return str(msg.get("content", "")).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run label OCR (or load JSON) and send it to a local Ollama model."
    )
    p.add_argument("image", nargs="?", help="Image path; OCR runs with same defaults as ocr.py.")
    p.add_argument(
        "--camera",
        action="store_true",
        help="Capture from the default webcam (Space to grab), then OCR + LLM.",
    )
    p.add_argument(
        "--ocr-json",
        type=Path,
        default=None,
        help="Skip OCR; use this JSON file (ocr.py --json format).",
    )
    p.add_argument(
        "--instructions",
        default="Describe this product and its most important features for someone who cannot see the label.",
        help="Extra user instructions for the model.",
    )
    p.add_argument(
        "--system",
        default=None,
        help="Override the system prompt (default: packaging / OCR cleanup).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API base URL.")
    p.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout seconds.")
    p.add_argument(
        "--dump-ocr",
        action="store_true",
        help="Print OCR JSON to stderr before calling the LLM.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    modes = sum(
        1 for x in (bool(args.image), bool(args.ocr_json), bool(args.camera)) if x
    )
    if modes == 0:
        print(
            "Provide an image path, --camera, or --ocr-json path/to.json\n"
            "  Example: python label_llm.py iv.jpeg\n"
            "  Example: python label_llm.py --camera",
            file=sys.stderr,
        )
        sys.exit(1)
    if modes > 1:
        print("Use only one of: image path, --camera, or --ocr-json.", file=sys.stderr)
        sys.exit(2)

    if args.ocr_json:
        ocr_payload = load_ocr_payload(args.ocr_json)
    else:
        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from ocr import LabelOcrConfig, capture_from_camera, run_label_ocr

        if args.camera:
            image = capture_from_camera()
        else:
            from PIL import Image

            path = Path(args.image).expanduser()
            if not path.is_file():
                print(f"Not a file: {path}", file=sys.stderr)
                sys.exit(1)
            image = Image.open(path).convert("RGB")
        ocr_payload = run_label_ocr(image, LabelOcrConfig())

    if args.dump_ocr:
        print(json.dumps(ocr_payload, indent=2), file=sys.stderr)

    user_body = {
        "ocr": ocr_payload,
        "instructions": args.instructions,
    }
    user_text = json.dumps(user_body, indent=2)
    system = args.system if args.system else DEFAULT_SYSTEM

    reply = ollama_chat(
        args.ollama_url.rstrip("/"),
        args.model,
        system,
        user_text,
        args.timeout,
    )
    print(reply)


if __name__ == "__main__":
    main()
