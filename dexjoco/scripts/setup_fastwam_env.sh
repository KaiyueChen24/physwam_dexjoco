#!/usr/bin/env bash
# FastWAM 训练/策略服务环境：Linux x86_64、Python 3.10、PyTorch CUDA 12.8。
set -Eeuo pipefail

DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd -- "$DEX_ROOT/../fastwam" && pwd)"
CONSTRAINTS="$DEX_ROOT/configs/environments/fastwam_constraints.txt"
ENV_NAME=fastwam
ENV_PREFIX=""
NAME_SET=false
PROXY_URL=""
SETUP_PROXY=false
PROXY_PORT=""
MIHOMO_BINARY=""
SUBSCRIPTION_FILE=""
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/physwam"
MODEL_DIR=""
MODEL_SOURCE=huggingface
GPU_INDEX=""
DRY_RUN=false

usage() {
    cat <<'EOF'
用法：bash dexjoco/scripts/setup_fastwam_env.sh [选项]
  --name NAME         [环境/名称] Conda 环境名，默认 fastwam；可复用已有 Python 3.10 环境
  --prefix PATH       [环境/路径] 使用完整环境路径，与显式 --name 互斥
  --proxy URL         [网络/代理] 使用已有 HTTP/HTTPS 代理，与 --setup-proxy 互斥
  --setup-proxy       [网络/安装] 安装或复用 Mihomo，并在后台启动本机代理
  --proxy-port PORT   [网络/端口] Mihomo 本机端口，默认 17890；需 --setup-proxy
  --mihomo-binary PATH [网络/离线安装] 使用已上传的 .gz/二进制；需 --setup-proxy
  --subscription-file PATH [网络/订阅] 使用已下载的订阅；需 --setup-proxy
  --cache-dir PATH    [缓存/AIFS] pip、Conda、HF、ModelScope、Torch 扩展与日志父目录
  --model-dir PATH    [权重/缓存] Wan 组件目录，默认同级 fastwam/checkpoints
  --model-source NAME [权重/下载] huggingface（默认）或 modelscope
  --check-gpu INDEX   [GPU/检查] 额外检查指定单张卡的 PyTorch BF16 运算；默认仅 CPU 检查
  --dry-run          [安装/计划] 只展示计划，不下载、不安装、不创建文件
  -h, --help         [帮助] 显示说明

需要 bash、curl 和可用网络；--setup-proxy 另需系统 python3（>=3.9）。未找到 Conda 时安装 Miniforge 到 ~/miniforge3。
现有环境必须使用 Python 3.10；不接受 base 环境作为安装目标。
首次准备代理时隐藏输入订阅链接；链接不写进仓库。GitHub 无法直连时需上传 Mihomo。
完成后 source ~/.config/physwam/fastwam-env.sh 加载当前服务器的代理和缓存设置。
EOF
}
die() { printf '错误：%s\n' "$*" >&2; exit 2; }
require_value() {
    [[ $# -ge 2 ]] || die "$1 缺少参数"
    [[ -n "$2" && "$2" != --* ]] || die "$1 缺少参数"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) require_value "$@"; ENV_NAME="$2"; NAME_SET=true; shift 2 ;;
        --prefix) require_value "$@"; ENV_PREFIX="$2"; shift 2 ;;
        --proxy) require_value "$@"; PROXY_URL="$2"; shift 2 ;;
        --setup-proxy) SETUP_PROXY=true; shift ;;
        --proxy-port) require_value "$@"; PROXY_PORT="$2"; shift 2 ;;
        --mihomo-binary) require_value "$@"; MIHOMO_BINARY="$2"; shift 2 ;;
        --subscription-file) require_value "$@"; SUBSCRIPTION_FILE="$2"; shift 2 ;;
        --cache-dir) require_value "$@"; CACHE_DIR="$2"; shift 2 ;;
        --model-dir) require_value "$@"; MODEL_DIR="$2"; shift 2 ;;
        --model-source) require_value "$@"; MODEL_SOURCE="$2"; shift 2 ;;
        --check-gpu) require_value "$@"; GPU_INDEX="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "未知选项：$1；使用 --help 查看说明" ;;
    esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die '目前仅支持 Linux x86_64'
[[ -f "$CONSTRAINTS" && -f "$FASTWAM_ROOT/pyproject.toml" && -f "$DEX_ROOT/openpi/packages/openpi-client/pyproject.toml" && -f "$DEX_ROOT/scripts/check_fastwam_env.py" ]] || die '源码不完整，请从完整仓库运行'
[[ -z "$ENV_PREFIX" || "$NAME_SET" == false ]] || die '--name 与 --prefix 不能同时指定'
[[ "$ENV_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ && "$ENV_NAME" != base ]] || die '环境名无效，或选择了 base'
[[ -z "$GPU_INDEX" || "$GPU_INDEX" =~ ^[0-9]+$ ]] || die '--check-gpu 只接受一个非负整数 GPU 编号'
[[ -z "$PROXY_URL" || "$PROXY_URL" =~ ^https?://[^[:space:]]+$ ]] || die '--proxy 需要 http:// 或 https:// 代理地址'
[[ "$SETUP_PROXY" != true || -z "$PROXY_URL" ]] || die '--setup-proxy 与 --proxy 不能同时指定'
[[ "$SETUP_PROXY" == true || -z "$PROXY_PORT$MIHOMO_BINARY$SUBSCRIPTION_FILE" ]] || die '端口、二进制、订阅选项需要 --setup-proxy'
if [[ -n "$PROXY_PORT" ]]; then
    [[ "$PROXY_PORT" =~ ^[0-9]{1,5}$ ]] || die '代理端口需要是 1024–65535 的整数'
    PROXY_PORT=$((10#$PROXY_PORT))
    (( PROXY_PORT >= 1024 && PROXY_PORT <= 65535 )) || die '代理端口范围应为 1024–65535'
fi
PROXY_PORT="${PROXY_PORT:-17890}"
for input_file in "$MIHOMO_BINARY" "$SUBSCRIPTION_FILE"; do
    [[ -z "$input_file" || -r "$input_file" && -f "$input_file" ]] || die 'Mihomo 二进制或订阅文件不存在/不可读'
done
[[ "$MODEL_SOURCE" == huggingface || "$MODEL_SOURCE" == modelscope ]] || die '--model-source 仅接受 huggingface 或 modelscope'
CACHE_DIR="$(readlink -m -- "$CACHE_DIR")"
MODEL_DIR="$(readlink -m -- "${MODEL_DIR:-$FASTWAM_ROOT/checkpoints}")"
if [[ -n "$ENV_PREFIX" ]]; then
    ENV_PREFIX="$(readlink -m -- "$ENV_PREFIX")"
    [[ "$ENV_PREFIX" != / && "$ENV_PREFIX" != "$HOME" ]] || die '不能将根目录或 home 作为环境目录'
    SELECTOR=(-p "$ENV_PREFIX")
else
    SELECTOR=(-n "$ENV_NAME")
fi

# 【环境发现】优先使用当前终端所属的 Conda；不需要预先 conda init。
CONDA_BIN="${CONDA_EXE:-}"
if [[ ! -x "$CONDA_BIN" ]]; then CONDA_BIN="$(type -P conda || true)"; fi
if [[ ! -x "$CONDA_BIN" ]]; then
    for candidate in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
    done
fi
if [[ "$DRY_RUN" == true ]]; then
    printf '源码：%s\n约束：%s\n' "$DEX_ROOT" "$CONSTRAINTS"
    printf 'Conda：%s\n目标环境：%s\n缓存：%s\n' "${CONDA_BIN:-未找到，将安装 ~/miniforge3}" "${ENV_PREFIX:-$ENV_NAME}" "$CACHE_DIR"
    if [[ "$SETUP_PROXY" == true ]]; then printf "网络：准备 Mihomo，回环端口 %s；首次可能隐藏输入订阅\n" "$PROXY_PORT"; elif [[ -n "$PROXY_URL" ]]; then echo '网络：使用指定代理（地址不写入日志）'; else echo '网络：继承当前终端代理设置'; fi
    echo '缓存：pip、Conda、Hugging Face、ModelScope、Torch 扩展；完成后生成 ~/.config/physwam/fastwam-env.sh'
    echo '步骤：Conda → Python 3.10/FFmpeg 7/编译工具 → PyTorch cu128 → DeepSpeed → FastWAM/轻量客户端 → pip check → CPU 视频/数据/配置检查'
    printf 'Wan 组件目录：%s\n模型下载源：%s\n' "$MODEL_DIR" "$MODEL_SOURCE"
    echo '仅配置环境；数据下载、模型初始化和训练另行执行'
    printf 'GPU 检查：%s\n' "${GPU_INDEX:-关闭}"
    exit 0
fi

# 【缓存/AIFS】路径由本机参数指定；脚本不会挂载 AIFS 或修改管理员配额。
umask 077
export PIP_CACHE_DIR="$CACHE_DIR/pip" CONDA_PKGS_DIRS="$CACHE_DIR/conda"
export HF_HOME="$CACHE_DIR/huggingface" HF_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/huggingface/datasets" MODELSCOPE_CACHE="$CACHE_DIR/modelscope"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions" TRITON_CACHE_DIR="$CACHE_DIR/triton"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_DIR" DIFFSYNTH_DOWNLOAD_SOURCE="$MODEL_SOURCE"
mkdir -p "$CACHE_DIR/logs" "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$MODEL_DIR" || die '无法写入缓存目录；请检查 AIFS 是否挂载及目录权限'
LOG_FILE="$(mktemp "$CACHE_DIR/logs/setup-fastwam.XXXXXX.log")"
VERSIONS_FILE="${LOG_FILE%.log}.versions.json"
exec > >(tee -a "$LOG_FILE") 2>&1
STEP='初始化'
INSTALLER_DIR=""
trap 'status=$?; printf "安装失败，阶段：%s；日志：%s\n" "$STEP" "$LOG_FILE" >&2; exit "$status"' ERR
trap 'if [[ -n "$INSTALLER_DIR" ]]; then rm -rf -- "$INSTALLER_DIR"; fi' EXIT
printf '安装日志：%s\n' "$LOG_FILE"

# 【网络/安装】先准备代理，再下载 Conda 和依赖；不覆盖已有 ~/.config/mihomo。
if [[ "$SETUP_PROXY" == true ]]; then
    STEP='准备 Mihomo 代理'
    command -v python3 >/dev/null || die '缺少系统 python3；请先安装 python3 或使用 --proxy'
    MIHOMO_ARGS=(--port "$PROXY_PORT")
    if [[ -n "$MIHOMO_BINARY" ]]; then MIHOMO_ARGS+=(--binary "$(readlink -m -- "$MIHOMO_BINARY")"); fi
    if [[ -n "$SUBSCRIPTION_FILE" ]]; then MIHOMO_ARGS+=(--subscription-file "$(readlink -m -- "$SUBSCRIPTION_FILE")"); fi
    python3 "$DEX_ROOT/scripts/setup_mihomo.py" "${MIHOMO_ARGS[@]}"
    PROXY_URL="http://127.0.0.1:$PROXY_PORT"
fi
if [[ -n "$PROXY_URL" ]]; then
    export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL"
    export all_proxy="$PROXY_URL" ALL_PROXY="$PROXY_URL"
    export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
    export NO_PROXY="$no_proxy"
fi

if [[ ! -x "$CONDA_BIN" ]]; then
    STEP='安装 Miniforge'
    command -v curl >/dev/null || die '缺少 curl，请先安装 curl 或自行安装 Conda'
    [[ ! -e "$HOME/miniforge3" ]] || die '~/miniforge3 已存在但缺少 conda，请检查目录后重试'
    INSTALLER_DIR="$(mktemp -d "${TMPDIR:-/tmp}/physwam-miniforge.XXXXXX")"
    curl -fL --retry 3 --connect-timeout 20 \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o "$INSTALLER_DIR/install.sh"
    bash "$INSTALLER_DIR/install.sh" -b -p "$HOME/miniforge3"
    CONDA_BIN="$HOME/miniforge3/bin/conda"
fi
STEP='检查 Conda 和目标环境'
CONDA_BASE="$("$CONDA_BIN" info --base)"
if [[ -n "$ENV_PREFIX" && "$ENV_PREFIX" == "$(readlink -m -- "$CONDA_BASE")" ]]; then
    die '不能将 Conda base 作为安装目标，请使用 --name fastwam 或其他 --prefix'
fi
run_python() { CUDA_VISIBLE_DEVICES='' "$CONDA_BIN" run --no-capture-output "${SELECTOR[@]}" python "$@"; }
run_gpu_python() { CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$CONDA_BIN" run --no-capture-output "${SELECTOR[@]}" python "$@"; }
printf 'Conda：%s\n' "$CONDA_BIN"
if "$CONDA_BIN" list "${SELECTOR[@]}" --json >/dev/null 2>&1; then
    # [已有环境] 不静默更换 Python 主次版本。
    run_python -c 'import sys; assert sys.version_info[:2] == (3, 10), "已有环境必须使用 Python 3.10；请另选 --name 或 --prefix"'
    TARGET_PREFIX="$(run_python -c 'import sys; print(sys.prefix)')"
    [[ "$(readlink -m -- "$TARGET_PREFIX")" != "$(readlink -m -- "$CONDA_BASE")" ]] || die '目标解析到了 Conda base'
    STEP='安装 FFmpeg 和编译工具'
    "$CONDA_BIN" install -y "${SELECTOR[@]}" --override-channels -c conda-forge \
        python=3.10 pip ffmpeg=7 c-compiler cxx-compiler make cmake ninja pkg-config
else
    STEP='创建 Python 3.10 环境'
    "$CONDA_BIN" create -y "${SELECTOR[@]}" --override-channels -c conda-forge \
        python=3.10 pip ffmpeg=7 c-compiler cxx-compiler make cmake ninja pkg-config
    TARGET_PREFIX="$(run_python -c 'import sys; print(sys.prefix)')"
fi

# 【依赖安装】先安装 CUDA 版 Torch，再在目标环境中构建 DeepSpeed 的 Python 包。
# 隐藏 GPU 并关闭预编译扩展，允许服务器 GPU 被占用时安装；不修改驱动或系统 CUDA。
STEP='安装构建工具'
run_python -m pip install -c "$CONSTRAINTS" setuptools wheel
STEP='安装 PyTorch CUDA 12.8'
run_python -m pip install -c "$CONSTRAINTS" \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
STEP='安装 DeepSpeed'
DS_BUILD_OPS=0 run_python -m pip install -c "$CONSTRAINTS" --no-build-isolation \
    deepspeed==0.18.7 --extra-index-url https://download.pytorch.org/whl/cu128
STEP='安装 FastWAM 与轻量客户端'
run_python -m pip install -c "$CONSTRAINTS" \
    -e "$FASTWAM_ROOT" -e "$DEX_ROOT/openpi/packages/openpi-client" \
    pyyaml websockets msgpack pytest \
    --extra-index-url https://download.pytorch.org/whl/cu128
STEP='pip check'
run_python -m pip check

# 【验证】CPU 检查不构造完整模型、不下载权重；GPU 运算仅在显式 --check-gpu 时执行。
STEP='CPU 导入、视频、数据、传输和配置检查'
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
run_python "$DEX_ROOT/scripts/check_fastwam_env.py" --mode cpu --strict-versions \
    --dexjoco-root "$DEX_ROOT" --constraints "$CONSTRAINTS" --report "$VERSIONS_FILE"
if [[ -n "$GPU_INDEX" ]]; then
    STEP='指定 GPU 的 PyTorch BF16 运算检查'
    run_gpu_python "$DEX_ROOT/scripts/check_fastwam_env.py" --mode gpu --strict-versions \
        --dexjoco-root "$DEX_ROOT" --constraints "$CONSTRAINTS" --report "$VERSIONS_FILE"
fi

# 【服务器设置】独立于 pi05-env.sh；shell 引号保护带空格或特殊字符的路径。
STEP='保存当前服务器的 FastWAM 设置'
mkdir -p "$HOME/.config/physwam"
ENV_FILE="$HOME/.config/physwam/fastwam-env.sh"
ENV_TEMP="$(mktemp "$HOME/.config/physwam/.fastwam-env.XXXXXX")"
{
    echo '# 当前服务器的 FastWAM 缓存/代理；由 setup_fastwam_env.sh 生成，激活环境后 source。'
    for env_key in PIP_CACHE_DIR CONDA_PKGS_DIRS HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE \
                   MODELSCOPE_CACHE TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR \
                   DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_DOWNLOAD_SOURCE; do
        printf 'export %s=%q\n' "$env_key" "${!env_key}"
    done
    if [[ -n "$PROXY_URL" ]]; then
        for env_key in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY; do
            printf 'export %s=%q\n' "$env_key" "${!env_key}"
        done
    fi
} > "$ENV_TEMP"
chmod 600 "$ENV_TEMP"
mv -f -- "$ENV_TEMP" "$ENV_FILE"

printf '\nFastWAM 环境配置完成。版本记录：%s\n' "$VERSIONS_FILE"
if [[ -z "$GPU_INDEX" ]]; then echo '本次通过 CPU 检查；GPU 运算可用 --check-gpu 编号检查，完整四卡训练另跑 fastwam_smoke。'; fi
printf '在当前终端加载环境：\nsource %q\nconda activate %q\n' "$CONDA_BASE/etc/profile.d/conda.sh" "$TARGET_PREFIX"
printf 'source %q\n' "$ENV_FILE"
echo '接下来准备任务数据，再执行 train_fastwam_group.py 的 prepare/inspect/initialize/train。'
