"""Test path setup for vendored agentspoker inside the hackathon repo."""

from __future__ import annotations

import sys
from pathlib import Path


AGENTSPOKER_ROOT = Path(__file__).resolve().parents[1]
if str(AGENTSPOKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTSPOKER_ROOT))
