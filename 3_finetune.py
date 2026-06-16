"""RoseMed Step 3/4: QLoRA fine-tuning with Unsloth."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from chat_format import build_conversation
from config import get_config

console = Console()
HEADER = """
══════ RoseMed Step 3/4: Fine-Tuning (QLoRA) ══════
"""


def _estimate_training_time(num_samples: int, epochs: int, batch_size: int, grad_accum: int) -> str:
    """Estimate training duration for an 80GB VRAM GPU."""
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, num_samples // effective_batch)
    total_steps = steps_per_epoch * epochs
    seconds_per_step = 12.0
    total_hours = (total_steps * seconds_per_step) / 3600
    return f"~{total_hours:.1f} hours ({total_steps} steps on 80GB VRAM GPU)"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    import json

    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _setup_wandb(cfg: Any) -> Optional[str]:
    """Configure WandB if API key is present."""
    if not cfg.wandb_api_key:
        return None
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    os.environ["WANDB_PROJECT"] = "rosemed"
    os.environ["WANDB_RUN_GROUP"] = "rosemed-27b-bg"
    return "rosemed-27b-bg"


def finetune() -> None:
    """Run QLoRA fine-tuning on the RoseMed dataset."""
    console.print(HEADER, style="bold cyan")

    cfg = get_config()
    model_path = cfg.model.local_model_path
    train_path = cfg.dataset.train_path
    val_path = cfg.dataset.val_path

    if not model_path.exists():
        console.print(
            "[red]ERROR:[/red] Base model not found at ./models/base-model\n"
            "Fix: Run python 1_download_model.py first.",
        )
        sys.exit(1)

    if not train_path.exists() or not val_path.exists():
        console.print(
            "[red]ERROR:[/red] Dataset not found.\n"
            "Fix: Run python 2_prepare_dataset.py first.",
        )
        sys.exit(1)

    train_samples = _load_jsonl(train_path)
    val_samples = _load_jsonl(val_path)

    est_time = _estimate_training_time(
        len(train_samples),
        cfg.training.num_train_epochs,
        cfg.training.per_device_train_batch_size,
        cfg.training.gradient_accumulation_steps,
    )
    console.print(f"Estimated training time: [yellow]{est_time}[/yellow]")
    console.print(
        f"Config: seq={cfg.training.max_seq_length}, "
        f"batch={cfg.training.per_device_train_batch_size}×"
        f"{cfg.training.gradient_accumulation_steps}, "
        f"lr={cfg.training.learning_rate}\n"
    )

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
        from trl import SFTTrainer
        from transformers import (
            TrainingArguments,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
        )
        from datasets import Dataset
        import torch
    except ImportError as exc:
        console.print(
            f"[red]ERROR:[/red] Missing dependency: {exc}\n"
            "Fix: Run bash setup.sh to install Unsloth and dependencies.",
        )
        sys.exit(1)

    if not cfg.model.chat_template:
        console.print(
            "[red]ERROR:[/red] CHAT_TEMPLATE is not set in .env.\n"
            "Fix: Add CHAT_TEMPLATE to your .env file (see HuggingFace model card).",
        )
        sys.exit(1)

    console.print("Loading base model (4-bit QLoRA)...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=cfg.training.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as exc:
        console.print(
            f"[red]ERROR:[/red] Failed to load model: {exc}\n"
            "Fix: Ensure model downloaded correctly and GPU has ≥ 80GB VRAM.",
        )
        sys.exit(1)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=cfg.model.chat_template,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=cfg.lora.target_modules,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    def formatting_func(examples: Dict[str, List[Any]]) -> List[str]:
        """Format examples using the RoseMed chat template."""
        texts: List[str] = []
        for i in range(len(examples["instruction"])):
            if "conversations" in examples and examples["conversations"][i]:
                convo = examples["conversations"][i]
            else:
                convo = build_conversation(
                    examples["instruction"][i],
                    examples["output"][i],
                    cfg.system_prompt,
                )
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return texts

    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    cfg.training.output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_name = _setup_wandb(cfg)
    report_to = "wandb" if wandb_run_name else "none"

    training_args = TrainingArguments(
        output_dir=str(cfg.training.output_dir),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=cfg.training.bf16 and is_bfloat16_supported(),
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.training.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=cfg.training.optim,
        report_to=report_to,
        run_name=wandb_run_name or "rosemed-27b-bg",
        seed=42,
        gradient_checkpointing=True,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    console.print("Starting QLoRA fine-tuning...")
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            formatting_func=formatting_func,
            max_seq_length=cfg.training.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            args=training_args,
            callbacks=callbacks,
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part=cfg.user_turn_start,
            response_part=cfg.model_turn_start,
        )

        train_result = trainer.train()
        eval_result = trainer.evaluate()

    except torch.cuda.OutOfMemoryError:
        console.print(
            "[red]ERROR:[/red] GPU out of memory.\n"
            "Fix: Ensure ≥ 80GB VRAM (A100/H100). Batch size is already minimal.",
        )
        sys.exit(1)
    except Exception as exc:
        console.print(
            f"[red]ERROR:[/red] Training failed: {exc}\n"
            "Fix: Check logs above for details.",
        )
        sys.exit(1)

    console.print("Saving LoRA adapter...")
    model.save_pretrained(str(cfg.training.output_dir))
    tokenizer.save_pretrained(str(cfg.training.output_dir))

    train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else "N/A"
    eval_loss = eval_result.get("eval_loss", "N/A")

    console.print(f"\n[green]✓[/green] Fine-tuning complete.")
    console.print(f"  Final train loss: {train_loss}")
    console.print(f"  Final eval loss:  {eval_loss}")
    console.print(f"  Adapter saved to: {cfg.training.output_dir}")
    console.print("\nNext step: [bold]python 4_merge_and_export.py[/bold]")


if __name__ == "__main__":
    finetune()
