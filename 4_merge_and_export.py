"""RoseMed Step 4/4: Merge LoRA adapter and export model."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from rich.console import Console

from config import get_config

console = Console()
HEADER = """
══════ RoseMed Step 4/4: Merge & Export ══════
"""

MODEL_CARD_TEMPLATE = """# RoseMed-27B-BG

## Model Description

**RoseMed-27B-BG** is a specialized 27-billion parameter Bulgarian medical AI assistant
built for the Bulgarian healthcare system. It delivers accurate medical information
in Bulgarian, aligned with НЗОК guidelines, the Bulgarian drug registry, and local
clinical protocols.

## Language

- Primary: Bulgarian (`bg`)
- Medical terminology: Bulgarian + English (`en`)

## Training Data

- 2,000 Bulgarian medical Q&A samples
- Categories: General Medicine, Cardiology, Neurology, Oncology, НЗОК/Health System,
  Emergency Medicine, Pharmacology, Pediatrics, Mental Health, Preventive Medicine
- All content tailored to Bulgarian healthcare context

## Architecture

- Parameters: 27B
- Context length: 4096 tokens
- Precision: bf16 (merged), Q4_K_M (GGUF export)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./rosemed-27b-bg", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("./rosemed-27b-bg")

messages = [
    {{"role": "system", "content": "Вие сте RoseMed, медицински асистент."}},
    {{"role": "user", "content": "Какви са симптомите на диабет?"}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### Example Queries (Bulgarian)

**Q:** Какви лекарства покрива НЗОК за хипертония?
**A:** НЗОК покрива Enalapril, Losartan, Amlodipine и други антихипертензивни...

**Q:** Какъв е вaccinaционният календар за деца?
**A:** Българският имунизационен календар включва BCG, DTP, MMR, HPV...

## Limitations

- Not intended for clinical decision-making without physician supervision
- May produce inaccurate information — always verify with a qualified doctor
- Does not replace emergency medical services (call 112)

## Disclaimer

⚕️ **Тази информация е само за справка. Консултирайте се с лекар.**

RoseMed-27B-BG is an informational tool only. It must not be used as a substitute
for professional medical advice, diagnosis, or treatment. Always seek the advice
of your physician or other qualified health provider.

## License

See LICENSE file for usage terms.
"""


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _dir_size(path: Path) -> int:
    """Calculate total directory size in bytes."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _sanitize_export_metadata(merged_path: Path, model_name: str) -> None:
    """Remove internal lineage references from exported model metadata."""
    config_path = merged_path / "config.json"
    if config_path.exists():
        config: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        config["_name_or_path"] = model_name
        for key in list(config.keys()):
            if any(token in key.lower() for token in ("base", "source", "parent")):
                config.pop(key, None)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    for meta_file in ("adapter_config.json", "trainer_state.json"):
        meta_path = merged_path / meta_file
        if meta_path.exists():
            meta_path.unlink()

    tokenizer_config = merged_path / "tokenizer_config.json"
    if tokenizer_config.exists():
        tok_cfg: Dict[str, Any] = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        tok_cfg.pop("name_or_path", None)
        tokenizer_config.write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")


def _export_gguf(
    merged_path: Path,
    gguf_path: Path,
    quantization: str,
) -> bool:
    """Export merged model to GGUF format using llama.cpp conversion."""
    console.print(f"Exporting GGUF ({quantization})...")

    convert_script = shutil.which("convert_hf_to_gguf.py")
    if convert_script is None:
        for candidate in (
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
        ):
            if candidate.exists():
                convert_script = str(candidate)
                break

    if convert_script is None:
        console.print(
            "[yellow]WARNING:[/yellow] GGUF conversion script not found.\n"
            "Install llama.cpp for GGUF export. Merged bf16 model is available.",
        )
        return False

    outtype = quantization.lower().replace("_", "")
    try:
        subprocess.run(
            [
                sys.executable,
                convert_script,
                str(merged_path),
                "--outfile",
                str(gguf_path),
                "--outtype",
                outtype,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return gguf_path.exists()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        console.print(
            f"[yellow]WARNING:[/yellow] GGUF export failed: {exc}\n"
            "Merged bf16 model saved successfully.",
        )
        return False


def merge_and_export() -> None:
    """Merge LoRA adapter with base model and export."""
    console.print(HEADER, style="bold cyan")

    cfg = get_config()
    base_path = cfg.model.local_model_path
    adapter_path = cfg.training.output_dir
    merged_path = cfg.export.merged_model_path
    gguf_path = cfg.export.gguf_path

    if not base_path.exists():
        console.print(
            "[red]ERROR:[/red] Base model not found.\n"
            "Fix: Run python 1_download_model.py",
        )
        sys.exit(1)

    if not adapter_path.exists():
        console.print(
            "[red]ERROR:[/red] LoRA adapter not found.\n"
            "Fix: Run python 3_finetune.py",
        )
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from peft import PeftModel
    except ImportError as exc:
        console.print(
            f"[red]ERROR:[/red] Unsloth not installed: {exc}\n"
            "Fix: Run bash setup.sh",
        )
        sys.exit(1)

    if not cfg.model.chat_template:
        console.print(
            "[red]ERROR:[/red] CHAT_TEMPLATE is not set in .env.\n"
            "Fix: Add CHAT_TEMPLATE to your .env file.",
        )
        sys.exit(1)

    console.print("Loading base model + LoRA adapter...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(base_path),
            max_seq_length=cfg.training.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=cfg.model.chat_template,
        )

        model = PeftModel.from_pretrained(model, str(adapter_path))
        console.print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

        FastLanguageModel.for_inference(model)

    except Exception as exc:
        console.print(
            f"[red]ERROR:[/red] Merge failed: {exc}\n"
            "Fix: Ensure adapter was saved correctly from fine-tuning step.",
        )
        sys.exit(1)

    merged_path.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving merged RoseMed model to {merged_path}...")

    try:
        if hasattr(model, "save_pretrained_merged"):
            model.save_pretrained_merged(
                str(merged_path),
                tokenizer,
                save_method="merged_16bit",
            )
        else:
            model.save_pretrained(str(merged_path), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_path))
    except OSError as exc:
        console.print(
            f"[red]ERROR:[/red] Failed to save merged model: {exc}\n"
            "Fix: Ensure sufficient disk space (~55 GB for bf16 27B).",
        )
        sys.exit(1)

    _sanitize_export_metadata(merged_path, cfg.model.output_model_name)

    model_card_path = merged_path / "README.md"
    model_card_path.write_text(MODEL_CARD_TEMPLATE, encoding="utf-8")
    console.print(f"Model card saved to {model_card_path}")

    gguf_success = _export_gguf(merged_path, gguf_path, cfg.export.gguf_quantization)

    console.print("\n[green]✓[/green] Export complete!\n")
    console.print("Exported files:")

    merged_size = _dir_size(merged_path)
    console.print(f"  Merged model: {merged_path} ({_format_size(merged_size)})")

    if gguf_success and gguf_path.exists():
        gguf_size = gguf_path.stat().st_size
        console.print(
            f"  GGUF ({cfg.export.gguf_quantization}): {gguf_path} ({_format_size(gguf_size)})"
        )
    else:
        console.print("  GGUF: [yellow]not exported[/yellow] (see warnings above)")

    console.print(f"\n  Model name: {cfg.model.output_model_name}")
    console.print("\n[bold green]Pipeline complete![/bold green] Start the API server:")
    console.print("  uvicorn server.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    merge_and_export()
