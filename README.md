# Smart Vision

Real-time scene segmentation using SegFormer and a live camera feed.

Colours in the preview window:

- **Green** -- walkable surfaces (floor, carpet, rug)
- **Red** -- obstacles (chair, table, sofa, bed, person)
- **Blue** -- walls

Navigation uses a simple per-frame pipeline:

1. **Walkable mask** -- floor / carpet / rug pixels in the lower path band
2. **Largest connected free-space region** -- biggest walkable blob (connected components)
3. **Centroid steering** -- steer toward that blob’s horizontal center (left / forward / right)
4. **Emergency stop zone** -- bottom-center rectangle; dense obstacles there force an immediate stop

The overlay also shows a yellow dot at the free-space centroid and a red box for the danger zone when triggered.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python -m project.main
```

This opens camera 0 at up to 30 FPS by default. Override with:

```bash
python -m project.main --camera-index 1 --max-fps 15
```

Press `q` in the preview window or `Ctrl+C` in the terminal to quit.

## Notes

- First run downloads the SegFormer model from Hugging Face (may take a minute).
- Uses **MPS** automatically on Apple Silicon (M1/M2/M3/M4) for faster inference.
- On macOS, allow camera access in **System Settings > Privacy & Security > Camera**.
