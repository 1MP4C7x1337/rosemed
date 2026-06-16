"""RoseMed Step 1/4: Download base model from HuggingFace Hub."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

load_dotenv()

console = Console()

HEADER = """
══════ RoseMed Step 1/4: Downloading Base Model ══════
"""


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _verify_download(model_dir: Path) -> bool:
    """Verify that key model files exist after download."""
    required_patterns = ["config.json", "tokenizer"]
    found_config = (model_dir / "config.json").exists()
    has_tokenizer = any(
        (model_dir / name).exists()
        for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    )
    has_weights = any(
        model_dir.glob("*.safetensors")
    ) or any(model_dir.glob("*.bin"))
    return found_config and has_tokenizer and has_weights


def download_base_model() -> None:
    """Authenticate and download the base model to local storage."""
    console.print(HEADER, style="bold cyan")

    hf_token = os.getenv("HF_TOKEN", "")
    base_model_id = os.getenv("BASE_MODEL_ID", "")
    local_path = Path("./models/base-model")

    if not hf_token or hf_token == "your_huggingface_token_here":
        console.print(
            "[red]ERROR:[/red] HF_TOKEN is missing or not configured.\n"
            "Fix: Set HF_TOKEN in your .env file with a valid HuggingFace token.",
            style="bold",
        )
        sys.exit(1)

    if not base_model_id:
        console.print(
            "[red]ERROR:[/red] BASE_MODEL_ID is not set in .env.\n"
            "Fix: Add BASE_MODEL_ID=your-model-id to .env",
            style="bold",
        )
        sys.exit(1)

    try:
        free_bytes = shutil.disk_usage(local_path.parent if local_path.parent.exists() else Path(".")).free
        min_required = 60 * 1024 ** 3  # ~60 GB
        if free_bytes < min_required:
            console.print(
                f"[yellow]WARNING:[/yellow] Low disk space: {_format_size(free_bytes)} free. "
                f"A 27B model requires ~60 GB.",
            )
    except OSError:
        pass

    console.print("Logging in to HuggingFace Hub...")
    try:
        login(token=hf_token, add_to_git_credential=False)
    except Exception as exc:
        console.print(
            f"[red]ERROR:[/red] HuggingFace login failed: {exc}\n"
            "Fix: Verify HF_TOKEN is valid at https://huggingface.co/settings/tokens",
        )
        sys.exit(1)

    local_path.mkdir(parents=True, exist_ok=True)
    console.print(f"Downloading base model to [green]{local_path}[/green]...")
    console.print("(This may take 30–90 minutes depending on connection speed.)\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading base model...", total=None)
            snapshot_download(
                repo_id=base_model_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=hf_token,
            )
            progress.update(task, completed=100, total=100)

    except OSError as exc:
        console.print(
            f"[red]ERROR:[/red] Disk error during download: {exc}\n"
            "Fix: Free disk space and re-run. Download supports resume.",
        )
        sys.exit(1)
    except Exception as exc:
        console.print(
            f"[red]ERROR:[/red] Download failed: {exc}\n"
            "Fix: Check network connection, HF_TOKEN permissions, and BASE_MODEL_ID.",
        )
        sys.exit(1)

    if not _verify_download(local_path):
        console.print(
            "[red]ERROR:[/red] Download verification failed — key files missing.\n"
            "Fix: Delete ./models/base-model and re-run this script.",
        )
        sys.exit(1)

    total_size = _dir_size(local_path)
    console.print(f"\n[green]✓[/green] Base model downloaded successfully.")
    console.print(f"  Location: {local_path.resolve()}")
    console.print(f"  Total size: {_format_size(total_size)}")
    console.print("\nNext step: [bold]python 2_prepare_dataset.py[/bold]")


if __name__ == "__main__":
    download_base_model()
