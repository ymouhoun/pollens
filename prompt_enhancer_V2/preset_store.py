"""Load prompt-enhancer presets without importing an inference backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


NEGATIVE_PATTERN = re.compile(
    r"<!--\s*NEGATIVE:\s*(.*?)\s*-->\s*",
    flags=re.DOTALL | re.IGNORECASE,
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, oversaturated, HDR, "
    "smooth skin, airbrushed, plastic skin, doll-like, watermark, text"
)


@dataclass(frozen=True)
class PresetCatalog:
    prompts: Mapping[str, str]
    negatives: Mapping[str, str]
    names: tuple[str, ...]

    def system_prompt(self, name: str) -> str:
        try:
            return self.prompts[name]
        except KeyError as exc:
            available = ", ".join(self.names)
            raise KeyError(
                f"Unknown style preset '{name}'. Available presets: {available}"
            ) from exc

    def negative_prompt(self, name: str) -> str:
        return self.negatives.get(name, DEFAULT_NEGATIVE_PROMPT)


def _preset_name(path: Path) -> str:
    name = path.stem
    prefix = "system_prompt_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def load_preset_catalog(directory: str | Path) -> PresetCatalog:
    preset_dir = Path(directory)
    if not preset_dir.is_dir():
        raise FileNotFoundError(f"Preset directory not found: {preset_dir}")

    prompts: dict[str, str] = {}
    negatives: dict[str, str] = {}

    for path in sorted(preset_dir.glob("*.md")):
        name = _preset_name(path)
        content = path.read_text(encoding="utf-8")
        match = NEGATIVE_PATTERN.search(content)
        if match:
            negatives[name] = " ".join(match.group(1).split())
            content = NEGATIVE_PATTERN.sub("", content)
        content = content.strip()
        if not content:
            raise ValueError(f"Preset '{name}' is empty: {path}")
        if name in prompts:
            raise ValueError(f"Duplicate preset name '{name}'")
        prompts[name] = content

    if not prompts:
        raise ValueError(f"No .md presets found in {preset_dir}")

    ordered = sorted(prompts)
    if "candid_raw" in ordered:
        ordered.remove("candid_raw")
        ordered.insert(0, "candid_raw")

    return PresetCatalog(
        prompts=MappingProxyType(prompts),
        negatives=MappingProxyType(negatives),
        names=tuple(ordered),
    )
