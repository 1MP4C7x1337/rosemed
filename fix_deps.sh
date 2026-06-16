#!/usr/bin/env bash
# Fix NumPy 2.x + TRL 1.x conflicts on Lightning.ai (conda cloudspace)
set -euo pipefail
cd "$(dirname "$0")"

echo "══════════════════════════════════════════"
echo "  RoseMed dependency fix (Lightning.ai)"
echo "══════════════════════════════════════════"

pip install --upgrade pip

echo "[1/4] Force NumPy 1.x (fixes Inf import error)..."
pip install "numpy==1.26.4" --force-reinstall --no-deps
pip install "scipy>=1.11.0,<1.28.0" --force-reinstall

echo "[2/4] Pin TRL for unsloth-zoo compatibility..."
pip install "trl==0.11.4" --force-reinstall

echo "[3/4] Install unsloth + unsloth-zoo..."
pip install unsloth-zoo --force-reinstall
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" --force-reinstall || pip install unsloth --force-reinstall

echo "[4/4] Verify..."
python3 << 'PYEOF'
import numpy as np
assert np.__version__.startswith("1."), f"NumPy must be 1.x, got {np.__version__}"
import scipy
import trl
import unsloth
import unsloth_zoo
print(f"numpy  {np.__version__}  OK")
print(f"scipy  {scipy.__version__}  OK")
print(f"trl    {trl.__version__}  OK")
print("unsloth + unsloth_zoo  OK")
print("")
print("Run: python3 3_finetune.py")
PYEOF
