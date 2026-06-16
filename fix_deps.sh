#!/usr/bin/env bash
# ONE-SHOT fix for Lightning.ai conda cloudspace — run: bash fix_deps.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "RoseMed: fixing all dependencies..."

pip install --upgrade pip -q

# 1. Rebuild numeric stack TOGETHER (fixes numpy.Inf + sklearn dtype errors)
pip uninstall -y scikit-learn scipy numpy 2>/dev/null || true
pip install --no-cache-dir "numpy==1.26.4"
pip install --no-cache-dir "scipy==1.11.4"
pip install --no-cache-dir "scikit-learn==1.4.2"

# 2. Pin ML stack compatible with unsloth-zoo
pip install --no-cache-dir \
  "transformers==4.44.2" \
  "peft==0.12.0" \
  "trl==0.11.4" \
  "datasets>=2.19.0" \
  "accelerate>=0.30.0" \
  "bitsandbytes>=0.43.0" \
  "sentencepiece" \
  "protobuf"

# 3. Unsloth last
pip install --no-cache-dir "unsloth-zoo"
pip install --no-cache-dir "unsloth==2024.6.7" || \
pip install --no-cache-dir "unsloth @ git+https://github.com/unslothai/unsloth.git"

echo ""
python3 -c "
import numpy as np
import sklearn
import unsloth
import unsloth_zoo
import trl
print('numpy', np.__version__)
print('sklearn', sklearn.__version__)
print('trl', trl.__version__)
print('')
print('SUCCESS — now run:  python3 3_finetune.py')
"
