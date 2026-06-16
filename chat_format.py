"""Shared chat formatting utilities for RoseMed training and inference."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import RoseMedConfig, get_config


def build_conversation(
    instruction: str,
    output: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build a multi-turn conversation dict for chat template application."""
    cfg = get_config()
    prompt = system_prompt or cfg.system_prompt
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]


def format_chat_text(
    instruction: str,
    output: str,
    system_prompt: Optional[str] = None,
    cfg: Optional[RoseMedConfig] = None,
) -> str:
    """Format a training sample using RoseMed chat turn markers."""
    config = cfg or get_config()
    prompt = system_prompt or config.system_prompt
    return (
        f"{config.user_turn_start}{prompt}\n\n{instruction}{config.turn_end}"
        f"{config.model_turn_start}{output}{config.turn_end}"
    )


def build_inference_prompt(
    user_message: str,
    history: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    cfg: Optional[RoseMedConfig] = None,
) -> str:
    """Build an inference prompt with history using RoseMed chat turn markers."""
    config = cfg or get_config()
    prompt = system_prompt or config.system_prompt
    parts: List[str] = []

    if not history:
        parts.append(f"{config.user_turn_start}{prompt}\n\n{user_message}{config.turn_end}")
    else:
        first_user = True
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                if first_user:
                    parts.append(
                        f"{config.user_turn_start}{prompt}\n\n{content}{config.turn_end}"
                    )
                    first_user = False
                else:
                    parts.append(f"{config.user_turn_start}{content}{config.turn_end}")
            elif role == "assistant":
                parts.append(f"{config.model_turn_start}{content}{config.turn_end}")

        parts.append(f"{config.user_turn_start}{user_message}{config.turn_end}")

    parts.append(config.model_turn_start)
    return "".join(parts)


def apply_tokenizer_template(
    tokenizer: Any,
    conversations: List[List[Dict[str, str]]],
) -> List[str]:
    """Apply the loaded tokenizer chat template to conversation batches."""
    texts: List[str] = []
    for convo in conversations:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return texts
