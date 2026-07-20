"""Pollens LLM Prompt Enhancer — GPU-only ComfyUI custom node."""

from .enhancer_node import LLMPromptEnhancer


NODE_CLASS_MAPPINGS = {
    "LLMPromptEnhancer": LLMPromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptEnhancer": "LLM Prompt Enhancer (GPU)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
