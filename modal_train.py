"""RoseMed v2 training on Modal.com — run from your laptop after crawl finishes on Lightning.

Setup (once):
  pip install modal
  modal setup
  modal secret create rosemed-secrets HF_TOKEN=xxx BASE_MODEL_ID=xxx CHAT_TEMPLATE=gemma-3

Upload data from Lightning (after crawl):
  modal volume create rosemed-data
  modal volume put rosemed-data ./v2_pack.tar.gz /v2_pack.tar.gz

Train:
  modal run modal_train.py

Download results:
  modal volume ls rosemed-models
  modal volume get rosemed-models /rosemed-27b-bg ./outputs/rosemed-27b-bg
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

import modal

APP_NAME = "rosemed-v2-train"
HOURS = 60 * 60

app = modal.App(APP_NAME)

# Persistent storage
DATA_VOL = modal.Volume.from_name("rosemed-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("rosemed-models", create_if_missing=True)

DATA_MOUNT = "/data"
MODEL_MOUNT = "/models"
OUTPUT_MOUNT = "/outputs"
WORK = "/root/rosemed"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.11.4",
        "scikit-learn==1.4.2",
        "torch==2.4.1",
        "transformers==4.44.2",
        "peft==0.12.0",
        "trl==0.11.4",
        "datasets>=2.19.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "protobuf",
        "huggingface_hub>=0.23.0",
        "python-dotenv>=1.0.0",
        "tqdm",
        "rich",
        "wandb",
    )
    .pip_install("unsloth-zoo")
    .pip_install("unsloth==2024.6.7")
    .env({"HF_HOME": f"{MODEL_MOUNT}/hf_cache"})
)

repo_mount = modal.Mount.from_local_dir(
    ".",
    remote_path=WORK,
    condition=lambda pth: not any(
        part in pth.replace("\\", "/")
        for part in ("/models/", "/outputs/", "/.git/", "/data/sources/crawl/")
    ),
)


def _run(cmd: list[str], cwd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(
    image=image,
    gpu="A100-80GB",  # 80GB for 27B QLoRA; ~$2.50/hr — fits $30 budget better than H100
    timeout=14 * HOURS,
    volumes={
        DATA_MOUNT: DATA_VOL,
        MODEL_MOUNT: MODEL_VOL,
        OUTPUT_MOUNT: MODEL_VOL,
    },
    secrets=[modal.Secret.from_name("rosemed-secrets")],
    mounts=[repo_mount],
)
def train_v2() -> None:
    """Download base model, prepare v2 dataset, fine-tune, merge."""
    os.chdir(WORK)

    # Write .env from Modal secrets
    env_path = Path(WORK) / ".env"
    env_path.write_text(
        f"HF_TOKEN={os.environ['HF_TOKEN']}\n"
        f"BASE_MODEL_ID={os.environ['BASE_MODEL_ID']}\n"
        f"CHAT_TEMPLATE={os.environ.get('CHAT_TEMPLATE', 'gemma-3')}\n",
        encoding="utf-8",
    )

    # Extract training pack uploaded from Lightning
    pack = Path(DATA_MOUNT) / "v2_pack.tar.gz"
    data_dir = Path(WORK) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if pack.exists():
        print(f"Extracting {pack} ...")
        with tarfile.open(pack, "r:gz") as tf:
            tf.extractall(WORK)
    else:
        print("WARNING: no v2_pack.tar.gz on volume — using existing data/ if present")

    full_train = data_dir / "rosemed_train_full.jsonl"
    train = data_dir / "rosemed_train.jsonl"
    if full_train.exists():
        train.write_bytes(full_train.read_bytes())
        lines = sum(1 for _ in train.open(encoding="utf-8"))
        print(f"Using v2 dataset: {lines} samples")
    elif not train.exists():
        raise FileNotFoundError(
            "No training data. Upload v2_pack.tar.gz:\n"
            "  modal volume put rosemed-data ./v2_pack.tar.gz /v2_pack.tar.gz"
        )

    # Symlink model cache to volume (persists across runs)
    models = Path(WORK) / "models"
    models.mkdir(exist_ok=True)
    hf_cache = Path(MODEL_MOUNT) / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    _run([sys.executable, "1_download_model.py"], WORK)
    _run([sys.executable, "3_finetune.py"], WORK)
    _run([sys.executable, "4_merge_and_export.py"], WORK)

    MODEL_VOL.commit()
    print("Done. Download merged model:")
    print("  modal volume get rosemed-models /rosemed-27b-bg ./outputs/rosemed-27b-bg")


@app.local_entrypoint()
def main():
    """Trigger v2 training on Modal."""
    train_v2.remote()


@app.local_entrypoint()
def upload_hint():
    """Print upload instructions."""
    print("""
After Lightning crawl finishes:

  cd ~/rosemed
  tar czf v2_pack.tar.gz data/rosemed_train_full.jsonl data/knowledge_base/
  # download v2_pack.tar.gz to your laptop, then:
  modal volume put rosemed-data ./v2_pack.tar.gz /v2_pack.tar.gz
  modal run modal_train.py
""")
