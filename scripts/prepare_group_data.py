"""Download only the chosen task group and its initialization weights."""
import argparse
import json

from group_common import ROOT, load_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("single_arm", "dual_arm", "all"), default="all")
    parser.add_argument("--data-revision", default="main")
    parser.add_argument("--model-revision", default="main")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data-only", action="store_true")
    mode.add_argument("--weights-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    groups = [load_group(g) for g in (("single_arm", "dual_arm") if args.group == "all" else (args.group,))]
    data_patterns = [f"dexjoco_lerobot_datasets/{task['dataset']}/*" for g in groups for task in g["tasks"]]
    model_patterns = ["pi05_base/*" if g["name"] == "single_arm" else "pi05_base_action_dim_44/*" for g in groups]
    plan = {"groups": [g["name"] for g in groups], "data_patterns": data_patterns, "model_patterns": model_patterns}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    if not args.weights_only:
        repository = "DexJoCo/DexJoCo-Datasets-LeRobot"
        revision = api.repo_info(repository, repo_type="dataset", revision=args.data_revision).sha
        snapshot_download(repository, repo_type="dataset", revision=revision, allow_patterns=data_patterns,
                          local_dir=ROOT / "datasets")
        path = ROOT / "datasets/download_manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {"tasks": {}}
        for group in groups:
            for task in group["tasks"]:
                manifest["tasks"][task["name"]] = {"repo_id": repository, "revision": revision}
        path.write_text(json.dumps(manifest, indent=2))
    if not args.data_only:
        repository = "DexJoCo/DexJoCo-Pi05"
        revision = api.repo_info(repository, revision=args.model_revision).sha
        snapshot_download(repository, revision=revision, allow_patterns=model_patterns,
                          local_dir=ROOT / "checkpoints")
        (ROOT / "checkpoints/download_manifest.json").write_text(json.dumps({
            "repo_id": repository, "revision": revision, "patterns": model_patterns}, indent=2))


if __name__ == "__main__":
    main()
