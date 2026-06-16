#!/usr/bin/env bash
# RoseMed-27B-BG — Environment setup script
# Platform-agnostic: RunPod, Vast.ai, Lambda Labs, Google Colab, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "══════════════════════════════════════════════════════════════"
echo "  RoseMed-27B-BG — Environment Setup"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Detect GPU provider ──────────────────────────────────────
PLATFORM="Unknown (generic GPU host)"

if [[ -n "${RUNPOD_POD_ID:-}" ]] || [[ -n "${RUNPOD_POD_HOSTNAME:-}" ]]; then
    PLATFORM="RunPod"
elif [[ -n "${VAST_CONTAINERLABEL:-}" ]] || [[ -f /etc/vast ]]; then
    PLATFORM="Vast.ai"
elif [[ -n "${LAMBDA_CLOUD_INSTANCE_ID:-}" ]]; then
    PLATFORM="Lambda Labs"
elif [[ -n "${COLAB_GPU:-}" ]] || [[ -n "${COLAB_RELEASE_TAG:-}" ]]; then
    PLATFORM="Google Colab"
elif [[ -n "${LIGHTNING_CLOUD_PROJECT_ID:-}" ]] || [[ -d "/teamspace/studios" ]]; then
    PLATFORM="Lightning.ai"
fi

echo "Detected platform: $PLATFORM"
echo ""

# ── Python 3.11+ (3.12 OK on Lightning.ai) ───────────────────
_select_python() {
    for candidate in python3.12 python3.11 python3; do
        if command -v "$candidate" &>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if ! PYTHON=$(_select_python); then
    echo "ERROR: Python 3 not found. Install Python 3.11+ and re-run."
    exit 1
fi

PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ]]; then
    if [[ "$PLATFORM" == "Lightning.ai" ]] || [[ -n "${LIGHTNING_CLOUD_PROJECT_ID:-}" ]]; then
        echo "ERROR: Python 3.11+ required (found ${PY_MAJOR}.${PY_MINOR})."
        exit 1
    fi
    echo "Python 3.11+ not found (found ${PY_MAJOR}.${PY_MINOR}). Attempting install..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq || true
        sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev || true
        PYTHON=$(_select_python)
    fi
fi

echo "Using Python: $($PYTHON --version)"
echo ""

# ── Virtual environment (optional but recommended) ───────────
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PIP="pip"
elif [[ -f ".venv/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
    PIP="pip"
else
    PIP="$PYTHON -m pip"
fi

echo "Upgrading pip..."
$PIP install --upgrade pip setuptools wheel -q

# ── Install requirements ─────────────────────────────────────
echo "Installing dependencies from requirements.txt..."
$PIP install -r requirements.txt -q

# ── Install Unsloth with auto-detected CUDA ──────────────────
echo "Detecting CUDA version for Unsloth..."
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || echo "12")
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
        CUDA_VERSION="cu124"
    elif [[ "$CUDA_MAJOR" -ge 11 ]]; then
        CUDA_VERSION="cu118"
    else
        CUDA_VERSION="cu124"
    fi
else
    echo "WARNING: nvidia-smi not found. Defaulting to cu124."
    CUDA_VERSION="cu124"
fi

echo "Installing latest Unsloth (CUDA: $CUDA_VERSION)..."
$PIP install --upgrade pip
$PIP install "unsloth @ git+https://github.com/unslothai/unsloth.git" -q 2>/dev/null || \
$PIP install --upgrade unsloth -q

# ── Create directories ───────────────────────────────────────
echo "Creating project directories..."
mkdir -p data outputs models logs

# ── Copy .env if missing ──────────────────────────────────────
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it with your tokens."
else
    echo ".env already exists — skipping."
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✓ RoseMed setup complete!"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Platform:  $PLATFORM"
echo "Python:    $($PYTHON --version)"
echo ""
echo "Next steps:"
echo "  1. Edit .env with HF_TOKEN, BASE_MODEL_ID, and CHAT_TEMPLATE"
echo "  2. python 1_download_model.py"
echo "  3. python 2_prepare_dataset.py"
echo "  4. python 3_finetune.py"
echo "  5. python 4_merge_and_export.py"
echo ""
echo "For inference:"
echo "  uvicorn server.main:app --host 0.0.0.0 --port 8000"
echo ""
