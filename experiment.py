"""Active independent protocol; previous runs remain immutable."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RUN_VERSION = "v5"
RUN_DIR = ROOT / "data" / RUN_VERSION
