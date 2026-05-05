"""
HeyCyan phone client — run this inside Termux on the Android phone.

Flow per import:
  1. Wait for the glasses' Wi-Fi Direct HTTP server (default 192.168.49.127) to
     answer /files/media.config — this happens once you tap "Import" in the
     HeyCyan app.
  2. Parse media.config, pick the latest .jpg, download it from /files/<name>.
  3. Upload it to the laptop's FastAPI server at /upload.
  4. Speak the OCR text the laptop returns via `termux-tts-speak` so the glasses
     (or whatever audio is active) read the label aloud.

By default the script loops: tap Import as many times as you like and each new
photo gets piped through. Pass --once for the original single-shot behavior.

One-time setup on the phone:
    pkg update && pkg upgrade
    pkg install python termux-api
    pip install requests
    # Also install the "Termux:API" companion app from F-Droid / Play Store.

Run:
    python phone_client.py --laptop http://192.168.0.31:8000
    python phone_client.py --once                                 # single import
    python phone_client.py --silent                               # no TTS
    python phone_client.py --glasses http://192.168.49.127 --laptop http://192.168.0.31:8000
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

DEFAULT_GLASSES_BASE = os.environ.get("HEYCYAN_GLASSES", "http://192.168.49.127")
DEFAULT_LAPTOP_BASE = os.environ.get("HEYCYAN_LAPTOP", "http://192.168.0.31:8000")
DEFAULT_OUT_DIR = Path(os.environ.get("HEYCYAN_OUT_DIR", "heycyan_images"))


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def speak(text: str, *, silent: bool) -> None:
    text = (text or "").strip()
    if not text or silent:
        return
    if have("termux-tts-speak"):
        try:
            subprocess.run(["termux-tts-speak", text], check=False)
            return
        except Exception as e:  # pragma: no cover - best effort
            print(f"[warn] termux-tts-speak failed: {e}", file=sys.stderr)
    print(f"[tts] {text}")


def wait_for_config(glasses_base: str, max_wait_sec: float = 120.0) -> str:
    print("Waiting for glasses Wi-Fi Direct server...")
    print("Now tap Import in the HeyCyan app.")

    start = time.time()
    url = f"{glasses_base.rstrip('/')}/files/media.config"

    while time.time() - start < max_wait_sec:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print("Connected to glasses.")
                return r.text
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    raise TimeoutError("Glasses server not reachable. Try Import again.")


def get_latest_jpg(config_text: str) -> str:
    files = re.findall(r"[\w\-]+\.jpg", config_text, flags=re.IGNORECASE)
    if not files:
        raise RuntimeError("No JPG found in media.config")
    return files[-1]


def download_image(glasses_base: str, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{glasses_base.rstrip('/')}/files/{filename}"
    print(f"Downloading {url}")

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    path = out_dir / filename
    path.write_bytes(r.content)
    print(f"Saved {path} ({len(r.content)} bytes)")
    return path


def upload_to_laptop(laptop_base: str, path: Path, timeout: float = 60.0) -> dict:
    url = f"{laptop_base.rstrip('/')}/upload"
    print(f"Uploading {path.name} to {url} ...")

    with path.open("rb") as f:
        files = {"file": (path.name, f, "image/jpeg")}
        r = requests.post(url, files=files, timeout=timeout)
    r.raise_for_status()

    try:
        payload = r.json()
    except ValueError:
        print("Upload complete (non-JSON response):", r.text)
        return {}
    return payload


def announce_payload(payload: dict, *, silent: bool) -> None:
    text = (payload.get("text") or "").strip()
    speech = (payload.get("speech") or "").strip()

    if text:
        print("--- Recognized text ---")
        print(text)
    else:
        print("[no text recognized]")

    speak(speech or text or "I could not read any text on the label.", silent=silent)


def process_once(args: argparse.Namespace, last_filename: str | None) -> str | None:
    """Run a single import → OCR → speak cycle. Returns the filename that was processed (or None)."""
    config = wait_for_config(args.glasses, max_wait_sec=args.wait)
    if args.dump_config:
        print("media.config:")
        print(config)

    latest = get_latest_jpg(config)
    if last_filename is not None and latest == last_filename:
        print(f"[skip] media.config still points at {latest}; waiting for a new import...")
        return last_filename

    try:
        img_path = download_image(args.glasses, latest, args.out_dir)
    except requests.RequestException as e:
        print(f"Glasses download failed: {e}", file=sys.stderr)
        speak("I could not download the photo from the glasses.", silent=args.silent)
        return last_filename

    try:
        payload = upload_to_laptop(args.laptop, img_path, timeout=args.timeout)
    except requests.RequestException as e:
        print(f"Laptop upload failed: {e}", file=sys.stderr)
        speak("I could not reach the server.", silent=args.silent)
        return latest

    announce_payload(payload, silent=args.silent)
    return latest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HeyCyan phone client: pull photo from glasses, upload to laptop, speak OCR result."
    )
    p.add_argument(
        "--glasses",
        default=DEFAULT_GLASSES_BASE,
        help=f"Glasses Wi-Fi Direct base URL (default: {DEFAULT_GLASSES_BASE}). "
        "Override with HEYCYAN_GLASSES.",
    )
    p.add_argument(
        "--laptop",
        default=DEFAULT_LAPTOP_BASE,
        help=f"Laptop FastAPI base URL (default: {DEFAULT_LAPTOP_BASE}). "
        "Override with HEYCYAN_LAPTOP.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Local folder to save downloaded photos (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--wait",
        type=float,
        default=120.0,
        help="Seconds to wait for the glasses server each cycle (default: 120).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for laptop upload (default: 60).",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Process a single import and exit (original behavior).",
    )
    p.add_argument(
        "--silent",
        action="store_true",
        help="Skip TTS; only print the recognized text.",
    )
    p.add_argument(
        "--dump-config",
        action="store_true",
        help="Print the raw media.config text from the glasses each cycle.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.once:
        try:
            process_once(args, last_filename=None)
        except (TimeoutError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            speak("I could not reach the glasses.", silent=args.silent)
            return 1
        return 0

    print("Loop mode: tap Import in the HeyCyan app each time. Ctrl-C to quit.")
    last: str | None = None
    try:
        while True:
            try:
                last = process_once(args, last_filename=last)
            except (TimeoutError, RuntimeError) as e:
                print(str(e), file=sys.stderr)
                speak("I could not reach the glasses.", silent=args.silent)
                # Brief pause before retrying so we don't spin if the glasses are off.
                time.sleep(2.0)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
