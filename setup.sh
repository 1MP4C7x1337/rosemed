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
fi

echo "Detected platform: $PLATFORM"
echo ""

# ── Python 3.11 ──────────────────────────────────────────────
if command -v python3.11 &>/dev/null; then
    PYTHON=python3.11
elif command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PY_VERSION" == "3.11" ]]; then
        PYTHON=python3
    else
        echo "Python 3.11 not found (found $PY_VERSION). Attempting install..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
            PYTHON=python3.11
        elif command -v yum &>/dev/null; then
            sudo yum install -y python3.11 python3.11-devel
            PYTHON=python3.11
        else
            echo "ERROR: Python 3.11 required. Install manually and re-run."
            exit 1
        fi
    fi
else
    echo "ERROR: Python not found. Install Python 3.11 and re-run."
    exit 1
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
