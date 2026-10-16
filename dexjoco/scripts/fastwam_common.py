"""Find the sibling FastWAM checkout without importing OpenPI/JAX/MuJoCo."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def resolve_launch_options(group, tasks, gpus, experiment):
    """CLI overrides YAML launch defaults; validate before touching data or CUDA."""
    group = group if group is not None else experiment.get("group")
    tasks = tasks if tasks is not None else experiment.get("tasks")
    gpus = gpus if gpus is not None else experiment.get("gpus", "0")
    if group not in ("single_arm", "dual_arm"):
        raise ValueError("Set group in the experiment YAML or pass --group")
    if tasks is not None and (not isinstance(tasks, list) or not tasks
                              or any(not isinstance(t, str) or not t.strip() for t in tasks)
                              or len(tasks) != len(set(tasks))):
        raise ValueError("tasks must be null (all tasks) or a nonempty list of unique task names")
    if not isinstance(gpus, str) or not gpus or any(not p.isdigit() for p in gpus.split(",")):
        raise ValueError("gpus must be a quoted comma-separated string of GPU IDs, e.g. '0,1,2,3'")
    if len(gpus.split(",")) != len(set(gpus.split(","))):
        raise ValueError("GPU IDs must be unique")
    return group, tasks, gpus


def add_fastwam(root=None):
    root = Path(root).expanduser().resolve() if root else ROOT.parent / "fastwam"
    if not (root / "experiments/dexjoco/config.py").is_file():
        raise FileNotFoundError(f"FastWAM integration missing: {root}; use --fastwam-root")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    return root
