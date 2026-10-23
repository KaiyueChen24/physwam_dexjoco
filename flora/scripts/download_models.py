"""Preview or explicitly download upstream-pinned model assets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("component", choices=["rewarder", "captioner"])
    p.add_argument("--execute", action="store_true", help="Download weights; without this flag only print revisions")
    args = p.parse_args()
    if args.component == "captioner":
        specs = json.loads((ROOT / "vendor/data_engine/batch_utils/models.json").read_text())
    else:
        spec = json.loads((ROOT / "vendor/sources.lock.json").read_text())["sources"]["rewarder"]
        specs = {"rewarder": {"repo": spec["repo_id"], "revision": spec["revision"]}}
    print(json.dumps(specs, indent=2))
    if not args.execute:
        return
    from huggingface_hub import snapshot_download

    if args.component == "rewarder":
        print(snapshot_download(specs["rewarder"]["repo"], revision=specs["rewarder"]["revision"]))
    else:
        sys.path.insert(0, str(ROOT / "vendor/data_engine"))
        from batch_utils.models import snapshot, validate_manifest

        for kind in ("tempseg", "captioner"):
            path = snapshot(kind)
            validate_manifest(path)
            print(path)
        spec = specs["tempseg_base"]
        print(snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=["*.safetensors", "*.json", "*.jinja"]))


if __name__ == "__main__":
    main()
