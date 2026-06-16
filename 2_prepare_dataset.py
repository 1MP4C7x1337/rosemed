"""RoseMed Step 2/4: Generate Bulgarian medical Q&A dataset."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from chat_format import build_conversation, format_chat_text
from config import get_config
from dataset_seeds import (
    AGE_CONTEXTS,
    CATEGORY_TARGETS,
    CITIES,
    LOCATION_CONTEXTS,
    PATIENT_NAMES,
    QUESTION_PREFIXES,
    SEED_LIBRARY,
)

console = Console()
HEADER = """
══════ RoseMed Step 2/4: Preparing Dataset ══════
"""


def _vary_instruction(seed: Dict[str, str], variant_idx: int) -> str:
    """Apply template variation to a seed instruction."""
    rng = random.Random(variant_idx * 7919 + hash(seed["instruction"]) % 10000)

    prefix = rng.choice(QUESTION_PREFIXES)
    location_ctx = rng.choice(LOCATION_CONTEXTS).format(city=rng.choice(CITIES))
    age_ctx = rng.choice(AGE_CONTEXTS)
    patient = rng.choice(PATIENT_NAMES)

    base = seed["instruction"]
    variations = [
        f"{prefix} {base}",
        f"{location_ctx}{base}",
        f"Пациент ({age_ctx}): {base}",
        f"За пациент {patient} от {rng.choice(CITIES)}: {base}",
        f"{base} (контекст: {age_ctx}, {rng.choice(CITIES)})",
        base,
    ]
    return rng.choice(variations)


def _vary_output(seed: Dict[str, str], variant_idx: int) -> str:
    """Apply template variation to a seed output."""
    rng = random.Random(variant_idx * 6271 + hash(seed["output"]) % 10000)
    city = rng.choice(CITIES)
    patient = rng.choice(PATIENT_NAMES)

    suffixes = [
        f"\n\nПрепоръчва се консултация с лекар в {city}.",
        f"\n\nЗа пациенти от {city} — проверете местните НЗОК процедури.",
        f"\n\nПациент {patient} трябва да обсъди това с личния си лекар.",
        f"\n\nИнформацията е съобразена с българските медицински протоколи.",
        "",
    ]
    return seed["output"] + rng.choice(suffixes)


def _format_chat_sample(
    instruction: str,
    output: str,
    system_prompt: str,
) -> str:
    """Format sample using RoseMed chat turn markers."""
    return format_chat_text(instruction, output, system_prompt)


def _generate_category_samples(
    category: str,
    target_count: int,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Generate samples for a category using seed expansion."""
    category_seeds = [s for s in SEED_LIBRARY if s["category"] == category]
    if not category_seeds:
        console.print(f"[yellow]WARNING:[/yellow] No seeds for category: {category}")
        return []

    samples: List[Dict[str, Any]] = []
    variant_idx = 0

    while len(samples) < target_count:
        seed = category_seeds[variant_idx % len(category_seeds)]
        instruction = _vary_instruction(seed, variant_idx)
        answer = _vary_output(seed, variant_idx)

        sample = {
            "instruction": instruction,
            "input": "",
            "output": answer,
            "category": category,
            "language": "bg",
            "conversations": build_conversation(instruction, answer, system_prompt),
            "text": _format_chat_sample(instruction, answer, system_prompt),
        }
        samples.append(sample)
        variant_idx += 1

    return samples[:target_count]


def _split_dataset(
    samples: List[Dict[str, Any]],
    val_split: float,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split samples into train and validation sets."""
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    val_count = int(len(shuffled) * val_split)
    val_set = shuffled[:val_count]
    train_set = shuffled[val_count:]
    return train_set, val_set


def _write_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    """Write samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _print_statistics(
    all_samples: List[Dict[str, Any]],
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
) -> None:
    """Print dataset statistics table."""
    table = Table(title="RoseMed Dataset Statistics")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")

    categories = sorted(set(s["category"] for s in all_samples))
    for cat in categories:
        cat_all = [s for s in all_samples if s["category"] == cat]
        cat_train = [s for s in train_samples if s["category"] == cat]
        cat_val = [s for s in val_samples if s["category"] == cat]
        table.add_row(cat, str(len(cat_all)), str(len(cat_train)), str(len(cat_val)))

    table.add_row("─" * 30, "─" * 5, "─" * 5, "─" * 5, style="dim")
    table.add_row(
        "TOTAL",
        str(len(all_samples)),
        str(len(train_samples)),
        str(len(val_samples)),
        style="bold green",
    )
    console.print(table)


def prepare_dataset() -> None:
    """Generate and save the RoseMed Bulgarian medical dataset."""
    console.print(HEADER, style="bold cyan")

    cfg = get_config()
    system_prompt = cfg.system_prompt

    console.print(f"Seed library: [green]{len(SEED_LIBRARY)}[/green] examples")
    console.print(f"Target samples: [green]{cfg.dataset.num_samples}[/green]\n")

    all_samples: List[Dict[str, Any]] = []

    with tqdm(total=sum(CATEGORY_TARGETS.values()), desc="Generating samples") as pbar:
        for category, target in CATEGORY_TARGETS.items():
            category_samples = _generate_category_samples(
                category, target, system_prompt
            )
            all_samples.extend(category_samples)
            pbar.update(len(category_samples))

    if len(all_samples) != cfg.dataset.num_samples:
        console.print(
            f"[yellow]WARNING:[/yellow] Generated {len(all_samples)} samples "
            f"(target: {cfg.dataset.num_samples})"
        )

    train_samples, val_samples = _split_dataset(
        all_samples, cfg.dataset.val_split
    )

    try:
        _write_jsonl(cfg.dataset.train_path, train_samples)
        _write_jsonl(cfg.dataset.val_path, val_samples)
        _write_jsonl(cfg.dataset.output_path, all_samples)
    except OSError as exc:
        console.print(
            f"[red]ERROR:[/red] Failed to write dataset: {exc}\n"
            "Fix: Check disk space and write permissions for ./data/",
        )
        sys.exit(1)

    console.print(f"\n[green]✓[/green] Dataset saved:")
    console.print(f"  Train: {cfg.dataset.train_path} ({len(train_samples)} samples)")
    console.print(f"  Val:   {cfg.dataset.val_path} ({len(val_samples)} samples)")
    console.print(f"  Full:  {cfg.dataset.output_path} ({len(all_samples)} samples)\n")

    _print_statistics(all_samples, train_samples, val_samples)
    console.print("\nNext step: [bold]python 3_finetune.py[/bold]")


if __name__ == "__main__":
    prepare_dataset()
