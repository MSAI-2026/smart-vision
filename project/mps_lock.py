"""
mps_lock.py -- Shared lock that serialises Metal command-buffer submissions.

PyTorch's MPS backend is not thread-safe: if two threads (e.g. the main loop
running YOLO and the seg worker running SegFormer) submit work concurrently,
Metal aborts with "Scheduled handler provided after commit call".

Every model forward pass that targets MPS must acquire this lock. CPU / CUDA
paths don't need it; callers gate the acquire on device type.
"""

import threading

MPS_LOCK = threading.Lock()
