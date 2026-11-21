"""Console safety: a Windows console (cp1252) cannot print an emoji from a stream title and would crash the run."""
from __future__ import annotations

import sys


def safe_streams() -> None:
    """Switch stdout/stderr to UTF-8 and replace anything unprintable instead of raising."""
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
