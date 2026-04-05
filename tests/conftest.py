"""
conftest.py — pytest bootstrap for MAD-CLEAN tests.

Adds the project root to sys.path so `import mad_clean` resolves to
mad_clean/ (the installed subpackage). No import shims needed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
