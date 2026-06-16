#!/usr/bin/env bash
# RoseMed — Lightning.ai one-command bootstrap
# Usage (paste entire block in Lightning Terminal):
#   curl -fsSL https://raw.githubusercontent.com/YOUR_USER/rosemed/main/bootstrap_lightning.sh | bash
# Or after cloning:
#   bash bootstrap_lightning.sh

set -euo pipefail

REPO_URL="${ROSEMED_REPO_URL:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "$HOME/rosemed")"

echo "══════════════════════════════════════════════════════════════"
echo "  RoseMed-27B-BG — Lightning.ai Bootstrap"
echo "══════════════════════════════════════════════════════════════"

# ── 1. Get code ──────────────────────────────────────────────
if [[ -f "$SCRIPT_DIR/setup.sh" ]]; then
    cd "$SCRIPT_DIR"
    echo "Using existing rosemed directory: $(pwd)"
elif [[ -n "$REPO_URL" ]]; then
    echo "Cloning from $REPO_URL ..."
    git clone "$REPO_URL" "$HOME/rosemed"
    cd "$HOME/rosemed"
else
    echo "ERROR: RoseMed code not found."
    echo ""
    echo "Set ROSEMED_REPO_URL and re-run, e.g.:"
    echo '  export ROSEMED_REPO_URL="https://github.com/YOUR_USER/rosemed.git"'
    echo "  curl -fsSL .../bootstrap_lightning.sh | bash"
    echo ""
    echo "Or clone manually first:"
    echo '  git clone https://github.com/YOUR_USER/rosemed.git ~/rosemed'
    echo "  cd ~/rosemed && bash bootstrap_lightning.sh"
    exit 1
fi

# ── 2. Write .env from Lightning Secrets / env vars ──────────
echo "Creating .env from environment secrets..."

missing=()
[[ -z "${HF_TOKEN:-}" ]]           && missing+=("HF_TOKEN")
[[ -z "${BASE_MODEL_ID:-}" ]]      && missing+=("BASE_MODEL_ID")
[[ -z "${CHAT_TEMPLATE:-}" ]]      && missing+=("CHAT_TEMPLATE")

cat > .env << EOF
HF_TOKEN=${HF_TOKEN:-}
BASE_MODEL_ID=${BASE_MODEL_ID:-}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-}
WANDB_API_KEY=${WANDB_API_KEY:-}
EOF

if [[ ${#missing[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: Missing secrets: ${missing[*]}"
    echo "Add them in Lightning → Settings → Secrets, then restart Studio"
    echo "or export them before running this script:"
    echo '  export HF_TOKEN="hf_..."'
    echo '  export BASE_MODEL_ID="..."'
    echo '  export CHAT_TEMPLATE="..."'
    echo ""
fi

# ── 3. Install dependencies ──────────────────────────────────
bash setup.sh

# ── 4. Run pipeline in tmux (survives disconnect) ─────────────
if command -v tmux &>/dev/null; then
    echo ""
    echo "Starting full pipeline in tmux session 'rosemed'..."
    tmux kill-session -t rosemed 2>/dev/null || true
    tmux new-session -d -s rosemed "cd $(pwd) && \
        python 1_download_model.py && \
        python 2_prepare_dataset.py && \
        python 3_finetune.py && \
        python 4_merge_and_export.py && \
        echo 'PIPELINE COMPLETE' | tee logs/pipeline_done.txt"
    echo ""
    echo "Pipeline running in background. Monitor with:"
    echo "  tmux attach -t rosemed"
    echo "  tail -f logs/* 2>/dev/null"
else
    echo ""
    echo "tmux not found. Run steps manually:"
    echo "  python 1_download_model.py"
    echo "  python 2_prepare_dataset.py"
    echo "  python 3_finetune.py"
    echo "  python 4_merge_and_export.py"
fi

echo ""
echo "Done. Reattach: tmux attach -t rosemed"
