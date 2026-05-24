"""
flow.py -- Warp a stale segmentation mask onto the current frame.

SegFormer-B0 runs at ~7-10 Hz on the worker thread, but the camera produces
frames at ~30 Hz. Between segmentation refreshes we estimate dense optical flow
on a downscaled grey frame (Farneback, ~2-5 ms) and remap the previous class
map so the floor mask doesn't visibly lag the camera.
"""

import cv2
import numpy as np

# Downscale factor for the flow computation. The class map is full-res but the
# flow itself is estimated at this resolution to keep latency tiny.
FLOW_SCALE = 0.25


def warp_class_map(
    class_map: np.ndarray,
    prev_frame_rgb: np.ndarray,
    curr_frame_rgb: np.ndarray,
) -> np.ndarray:
    """
    Remap `class_map` from `prev_frame` coords into `curr_frame` coords using
    dense optical flow. Returns the warped class map (same shape as class_map).
    """
    if class_map is None or prev_frame_rgb is None or curr_frame_rgb is None:
        return class_map

    h, w = class_map.shape[:2]

    # Compute flow at low resolution.
    sw, sh = max(1, int(w * FLOW_SCALE)), max(1, int(h * FLOW_SCALE))
    prev_small = cv2.resize(prev_frame_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    curr_small = cv2.resize(curr_frame_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    prev_grey = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
    curr_grey = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY)

    flow_small = cv2.calcOpticalFlowFarneback(
        prev_grey, curr_grey, None,
        pyr_scale=0.5, levels=2, winsize=13,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
    )

    # Up-sample flow to full resolution and rescale magnitudes accordingly.
    flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= w / sw
    flow[..., 1] *= h / sh

    # cv2.remap maps OUTPUT(x,y) <- INPUT(map_x(x,y), map_y(x,y)).
    # We want curr(x,y) = prev(x - dx, y - dy), so subtract flow.
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]

    warped = cv2.remap(
        class_map.astype(np.int32),
        map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped.astype(np.int32)
