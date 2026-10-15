"""Find the sibling FastWAM checkout without importing OpenPI/JAX/MuJoCo."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def add_fastwam(root=None):
    root = Path(root).expanduser().resolve() if root else ROOT.parent / "fastwam"
    if not (root / "experiments/dexjoco/config.py").is_file():
        raise FileNotFoundError(f"FastWAM integration missing: {root}; use --fastwam-root")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    return root
