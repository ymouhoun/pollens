"""Pure prompt construction and output cleanup for the enhancer."""

from __future__ import annotations

import re


_PREAMBLE_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:Here|Sure|Okay|Let me|I'll|I will|The enhanced|Enhanced|Below|Output)[^\n]*\n+",
        r"^(?:Certainly|Of course|Alright|Great|Now)[^\n]*\n+",
        r"^OUTPUT:\s*",
        r"^(?:Photograph prompt|Enhanced prompt|Prompt):\s*\n*",
    )
)


def build_user_prompt(prompt: str, prompt_input: str = "") -> str:
    prompt = (prompt or "").strip()
    prompt_input = (prompt_input or "").strip()
    if prompt_input:
        parts = [prompt_input]
        if prompt:
            parts.append(f"Additional instructions: {prompt}")
        return "\n\n".join(parts)
    return prompt


def prepare_system_prompt(system_prompt: str, model_name: str) -> str:
    result = system_prompt.strip()
    normalized = model_name.lower()
    if "qwen3" in normalized or "qwen-3" in normalized:
        result += "\n\n/no_think"
    return result


def required_context(
    system_prompt: str,
    user_prompt: str,
    requested_context: int,
    max_tokens: int,
    maximum_context: int = 32768,
) -> int:
    estimated_input_tokens = (len(system_prompt) + len(user_prompt)) // 3 + 64
    needed = estimated_input_tokens + max_tokens
    if needed <= requested_context:
        return requested_context
    proposed = ((needed // 1024) + 1) * 1024
    if proposed > maximum_context:
        raise ValueError(
            "Preset and prompt exceed the supported context window "
            f"(~{needed} tokens required, maximum {maximum_context})."
        )
    return proposed


def clean_model_output(value: str) -> str:
    result = (value or "").strip()
    result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.DOTALL).strip()
    result = re.sub(r"<think>.*", "", result, flags=re.DOTALL).strip()

    for pattern in _PREAMBLE_PATTERNS:
        result = pattern.sub("", result).strip()

    if len(result) > 2 and (
        (result.startswith('"') and result.endswith('"'))
        or (result.startswith("'") and result.endswith("'"))
    ):
        result = result[1:-1].strip()

    if result.lower().startswith(("okay", "let me", "i need", "first", "the user")):
        paragraphs = [part.strip() for part in result.split("\n\n") if part.strip()]
        if len(paragraphs) > 1:
            result = paragraphs[-1]
    return result


def apply_prompt_prefix(value: str, prefix: str = "") -> str:
    normalized = (prefix or "").strip().rstrip(",")
    return f"{normalized}, {value}" if normalized else value
