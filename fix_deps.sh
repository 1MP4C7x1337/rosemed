#!/usr/bin/env bash
# Fix broken Unsloth/TRL/NumPy versions on Lightning.ai
set -euo pipefail
cd "$(dirname "$0")"

echo "Fixing RoseMed training dependencies..."

pip install --upgrade pip

# NumPy 2.x breaks scipy + unsloth_zoo (Inf import error)
pip install "numpy>=1.24.0,<2.0.0" "scipy>=1.11.0,<1.28.0"

# TRL 1.x breaks unsloth-zoo — must stay below 0.16
pip install "trl>=0.11.0,<0.16.0" transformers datasets accelerate peft bitsandbytes

pip install unsloth-zoo
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" || pip install unsloth

python3 -c "
import numpy as np
import unsloth
import unsloth_zoo
import trl
print('numpy', np.__version__)
print('trl', trl.__version__)
print('All OK — run: python3 3_finetune.py')
"
