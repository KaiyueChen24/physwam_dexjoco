#!/usr/bin/env bash
# Build a dedicated runtime. Never installs into the existing FastWAM/pi05 envs.
set -euo pipefail
flora_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
component="${1:-}"
if [[ "$component" != "rewarder" && "$component" != "captioner" ]]; then
  echo "Usage: bash flora/scripts/setup_env.sh rewarder|captioner [--dry-run]" >&2
  exit 2
fi
dry_run=false
if [[ "${2:-}" == "--dry-run" ]]; then dry_run=true;
elif [[ -n "${2:-}" ]]; then echo "Unknown option: $2" >&2; exit 2; fi
if (( $# > 2 )); then echo "Too many arguments" >&2; exit 2; fi
flora_python="${FLORA_PYTHON:-python3.12}"
flora_env="$flora_root/.venv-$component"
execute() {
  printf '%q ' "$@"; printf '\n'
  if ! $dry_run; then "$@"; fi
}
if ! $dry_run; then
  "$flora_python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Official FLORA examples use Python 3.12"'
fi
execute "$flora_python" -m venv "$flora_env"
execute "$flora_env/bin/python" -m pip install --upgrade pip
if [[ "$component" == captioner ]]; then
  execute "$flora_env/bin/python" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  execute "$flora_env/bin/python" -m pip install -r "$flora_root/vendor/data_engine/batch_utils/requirements.txt"
else
  execute "$flora_env/bin/python" -m pip install -r "$flora_root/vendor/rewarder/batch_utils/requirements.txt"
fi
execute "$flora_env/bin/python" -m pip install -e "$flora_root[data]"
execute "$flora_env/bin/python" -m pip check
echo "Runtime: $flora_env/bin/python"
echo 'Model weights are downloaded only by download_models.py --execute or an inference run.'
