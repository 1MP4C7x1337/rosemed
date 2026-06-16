"""Central configuration for the RoseMed-27B-BG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Model paths and identifiers."""

    base_model_id: str = field(default_factory=lambda: os.getenv("BASE_MODEL_ID", ""))
    output_model_name: str = "rosemed-27b-bg"
    local_model_path: Path = Path("./models/base-model")
    chat_template: str = field(default_factory=lambda: os.getenv("CHAT_TEMPLATE", ""))


@dataclass
class LoRAConfig:
    """LoRA fine-tuning hyperparameters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    learning_rate: float = 1e-4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    output_dir: Path = Path("./outputs/lora_adapter")
    optim: str = "adamw_8bit"


@dataclass
class DatasetConfig:
    """Dataset generation and paths."""

    output_path: Path = Path("./data/rosemed_bulgarian_medical.jsonl")
    train_path: Path = Path("./data/rosemed_train.jsonl")
    val_path: Path = Path("./data/rosemed_val.jsonl")
    num_samples: int = 2000
    val_split: float = 0.1


@dataclass
class InferenceConfig:
    """API server inference settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class ExportConfig:
    """Model export paths and quantization."""

    merged_model_path: Path = Path("./outputs/rosemed-27b-bg")
    gguf_path: Path = Path("./outputs/rosemed-27b-bg-Q4_K_M.gguf")
    gguf_quantization: str = "Q4_K_M"


@dataclass
class RoseMedConfig:
    """Top-level configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    wandb_api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))

    system_prompt: str = (
        "Вие сте RoseMed, високоспециализиран български медицински AI асистент, "
        "обучен върху българската здравна система. Отговаряте на български език, "
        "съобразено с НЗОК, български медицински протоколи и регистрирани лекарства. "
        "Винаги препоръчвайте консултация с лекар."
    )

    medical_disclaimer: str = (
        "⚕️ Тази информация е само за справка. Консултирайте се с лекар."
    )

    user_turn_start: str = "<start_of_turn>user\n"
    model_turn_start: str = "<start_of_turn>model\n"
    turn_end: str = "<end_of_turn>\n"


def get_config() -> RoseMedConfig:
    """Return the global RoseMed configuration."""
    return RoseMedConfig()
