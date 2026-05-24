#!/usr/bin/env python3
"""
run.py -- Launch Smart Vision from the repository root.

Usage:
    python run.py
    python run.py --camera-index 1 --max-fps 15
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports work when invoked as `python run.py` from any cwd.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from project.main import main

if __name__ == "__main__":
    main()
