"""Project-wide Python startup settings.

Python imports this module automatically when commands are run from the
repository.  Keeping Matplotlib's cache inside the writable project prevents
home-directory permission warnings in restricted and containerized runs.
"""

from __future__ import annotations

import os
from pathlib import Path


_matplotlib_cache = Path(__file__).resolve().parent / "logs" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
