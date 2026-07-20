"""ComfyUI adapter for the GPU-only prompt enhancer."""

from __future__ import annotations

import os
from pathlib import Path
import time

import folder_paths

from .gpu_runtime import GPUWorkerError, generate_on_gpu
from .preset_store import DEFAULT_NEGATIVE_PROMPT, load_preset_catalog
from .prompt_logic import (
    apply_prompt_prefix,
    build_user_prompt,
    clean_model_output,
    prepare_system_prompt,
    required_context,
)


PLUGIN_DIR = Path(__file__).resolve().parent
PRESETS_DIR = PLUGIN_DIR / "presets"
LLM_DIR = Path(folder_paths.models_dir) / "llm_gguf"
LLM_DIR.mkdir(parents=True, exist_ok=True)

if "llm_gguf" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["llm_gguf"] = ([str(LLM_DIR)], {".gguf"})

PRESETS = load_preset_catalog(PRESETS_DIR)


class LLMPromptEnhancer:
    """Enhance a text prompt through an isolated, persistent GPU worker."""

    CATEGORY = "conditioning/llm"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt")
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("llm_gguf") or ["no_models_found.gguf"]
        default_model = models[0]
        for name in models:
            if name.lower() == "qwen3-8b-q8_0.gguf":
                default_model = name
                break

        return {
            "required": {
                "model": (models, {"default": default_model}),
                "style_preset": (list(PRESETS.names), {"default": PRESETS.names[0]}),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Short scene description to enhance...",
                    },
                ),
            },
            "optional": {
                "prompt_input": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": "Connected text becomes the scene description; prompt adds instructions.",
                    },
                ),
                "system_prompt_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Leave empty to use the selected preset.",
                    },
                ),
                "prompt_prefix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional LoRA trigger token(s).",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 50, "max": 2048, "step": 50},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05},
                ),
                "n_ctx": (
                    "INT",
                    {"default": 8192, "min": 512, "max": 32768, "step": 512},
                ),
                "n_gpu_layers": (
                    "INT",
                    {"default": 99, "min": 1, "max": 100, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1},
                ),
            },
        }

    def enhance(
        self,
        model,
        style_preset,
        prompt,
        prompt_input="",
        system_prompt_override="",
        prompt_prefix="",
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        repeat_penalty=1.1,
        n_ctx=8192,
        n_gpu_layers=99,
        seed=0,
    ):
        if int(n_gpu_layers) <= 0:
            raise RuntimeError(
                "LLM Prompt Enhancer requires GPU offload; n_gpu_layers must be greater than zero."
            )

        user_prompt = build_user_prompt(prompt, prompt_input)
        negative = PRESETS.negative_prompt(style_preset)
        if not user_prompt:
            return ("", negative)

        if system_prompt_override and system_prompt_override.strip():
            system_prompt = system_prompt_override.strip()
            negative = DEFAULT_NEGATIVE_PROMPT
        else:
            system_prompt = PRESETS.system_prompt(style_preset)
        system_prompt = prepare_system_prompt(system_prompt, model)
        effective_context = required_context(
            system_prompt,
            user_prompt,
            int(n_ctx),
            int(max_tokens),
        )
        if effective_context != int(n_ctx):
            print(
                f"[LLM Enhancer] Context raised from {n_ctx} to {effective_context} "
                f"for preset '{style_preset}'"
            )

        model_path = folder_paths.get_full_path("llm_gguf", model)
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"GGUF model '{model}' not found in {LLM_DIR}")
        model_path = os.fspath(model_path)

        payload = {
            "model_path": model_path,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "n_ctx": effective_context,
            "n_gpu_layers": int(n_gpu_layers),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "repeat_penalty": float(repeat_penalty),
            "seed": int(seed) if int(seed) > 0 else None,
        }

        print(
            f"[LLM Enhancer] GPU generation: model={model}, preset={style_preset}, "
            f"ctx={effective_context}, max_tokens={max_tokens}"
        )
        try:
            response = generate_on_gpu(payload)
        except GPUWorkerError as exc:
            raise RuntimeError(
                "LLM Prompt Enhancer GPU inference failed. CPU fallback is disabled: "
                f"{exc}"
            ) from exc

        result = clean_model_output(response.get("content", ""))
        if not result:
            raise RuntimeError("LLM Prompt Enhancer returned an empty GPU result")
        result = apply_prompt_prefix(result, prompt_prefix)
        print(f"[LLM Enhancer] Output ready ({len(result)} chars)")
        return {
            "ui": {
                "text": [result],
                "negative_prompt": [negative],
                "style_preset": [style_preset],
            },
            "result": (result, negative),
        }
