#!/usr/bin/env python3
"""JSON-lines GPU inference worker. This process may safely absorb native aborts."""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any


SUPPORTED_GPU_ARCHITECTURES = {
    "9.0": "90",
    "10.0": "100",
    "12.0": "120",
}
CERTIFIED_LLAMA_CPP_VERSION = "0.3.34"
MINIMUM_DRIVER_VERSION = (570, 124, 6)


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for component in value.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_backend_version(actual_version: str) -> tuple[bool, str]:
    """Return certification status and enforce an exact pin only on demand."""
    expected_version = os.environ.get(
        "LLM_ENHANCER_LLAMA_CPP_VERSION", CERTIFIED_LLAMA_CPP_VERSION
    )
    matches = actual_version == expected_version
    strict = os.environ.get("LLM_ENHANCER_STRICT_BACKEND_VERSION") == "1"
    allow_uncertified = os.environ.get("LLM_ENHANCER_ALLOW_UNCERTIFIED_BUILD") == "1"
    if not matches and strict and not allow_uncertified:
        raise RuntimeError(
            f"llama-cpp-python {actual_version} is not the certified version "
            f"{expected_version}; rebuild the Docker image"
        )
    return matches, expected_version


def detect_gpu() -> dict[str, str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty; GPU inference is mandatory")
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,compute_cap,driver_version",
        "--format=csv,noheader",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"NVIDIA GPU detection failed: {exc}") from exc

    devices: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",", 4)]
        if len(fields) == 5:
            devices.append(
                dict(zip(("index", "uuid", "name", "compute_cap", "driver"), fields))
            )
    if not devices:
        raise RuntimeError("nvidia-smi reported no NVIDIA GPU")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    selected = devices[0]
    if visible:
        for device in devices:
            if visible in {device["index"], device["uuid"]} or device["uuid"].startswith(visible):
                selected = device
                break

    capability = selected["compute_cap"]
    if capability not in SUPPORTED_GPU_ARCHITECTURES:
        supported = ", ".join(f"sm_{value}" for value in SUPPORTED_GPU_ARCHITECTURES.values())
        raise RuntimeError(
            f"Unsupported GPU compute capability {capability} ({selected['name']}); "
            f"certified targets are {supported}"
        )
    if version_tuple(selected["driver"]) < MINIMUM_DRIVER_VERSION:
        required = ".".join(str(part) for part in MINIMUM_DRIVER_VERSION)
        raise RuntimeError(
            f"NVIDIA driver {selected['driver']} is below the certified minimum {required}"
        )
    selected["sm"] = SUPPORTED_GPU_ARCHITECTURES[capability]
    return selected


class ModelCache:
    def __init__(self, llama_class: Any) -> None:
        self._llama_class = llama_class
        self._model: Any = None
        self._model_path: str | None = None
        self._n_ctx = 0
        self._n_gpu_layers = 0

    def _close(self) -> None:
        if self._model is None:
            return
        close = getattr(self._model, "close", None)
        if callable(close):
            close()
        self._model = None
        self._model_path = None
        self._n_ctx = 0
        self._n_gpu_layers = 0
        gc.collect()

    def get(self, request: dict[str, Any]) -> Any:
        model_path = str(Path(request["model_path"]).resolve())
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")
        n_ctx = int(request["n_ctx"])
        n_gpu_layers = int(request["n_gpu_layers"])
        if n_gpu_layers <= 0:
            raise RuntimeError("n_gpu_layers must be greater than zero in GPU-only mode")
        if (
            self._model is not None
            and self._model_path == model_path
            and self._n_ctx >= n_ctx
            and self._n_gpu_layers == n_gpu_layers
        ):
            return self._model

        self._close()
        threads = max(4, multiprocessing.cpu_count() // 2)
        self._model = self._llama_class(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=threads,
            n_threads_batch=threads,
            flash_attn=True,
            verbose=False,
        )
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        return self._model

    def close(self) -> None:
        self._close()


def generate(cache: ModelCache, request: dict[str, Any]) -> str:
    model = cache.get(request)
    kwargs: dict[str, Any] = {
        "messages": request["messages"],
        "temperature": float(request["temperature"]),
        "top_p": float(request["top_p"]),
        "max_tokens": int(request["max_tokens"]),
        "repeat_penalty": float(request["repeat_penalty"]),
    }
    seed = request.get("seed")
    if seed is not None:
        kwargs["seed"] = int(seed)
    try:
        response = model.create_chat_completion(**kwargs)
    except Exception:
        # A failed CUDA execution may leave a context unusable. Force a clean
        # reload on the next request instead of reusing uncertain GPU state.
        cache.close()
        raise
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise RuntimeError("LLM response did not contain text content")
    return content


def main() -> int:
    try:
        gpu = detect_gpu()
        import llama_cpp
        from llama_cpp import Llama

        actual_version = getattr(llama_cpp, "__version__", "unknown")
        certified_version, expected_version = check_backend_version(actual_version)
        if not certified_version:
            print(
                f"WARNING: llama-cpp-python {actual_version} differs from the "
                f"certified {expected_version}; continuing with a real GPU load test",
                file=sys.stderr,
                flush=True,
            )
        supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if not callable(supports_gpu) or not supports_gpu():
            raise RuntimeError(
                "llama-cpp-python was not compiled with GPU offload support; CPU fallback is disabled"
            )
        cache = ModelCache(Llama)
        emit(
            {
                "type": "ready",
                "pid": os.getpid(),
                "gpu_name": gpu["name"],
                "compute_capability": gpu["compute_cap"],
                "sm": gpu["sm"],
                "driver_version": gpu["driver"],
                "llama_cpp_version": actual_version,
                "backend_version_certified": certified_version,
            }
        )
    except BaseException as exc:
        emit({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    try:
        for line in sys.stdin:
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                request_type = request.get("type")
                if request_type == "shutdown":
                    break
                if request_type == "ping":
                    emit({"type": "pong", "request_id": request.get("request_id")})
                    continue
                if request_type != "generate":
                    raise ValueError(f"Unsupported request type: {request_type}")
                result = generate(cache, request)
                emit(
                    {
                        "type": "result",
                        "request_id": request.get("request_id"),
                        "content": result,
                    }
                )
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit(
                    {
                        "type": "error",
                        "request_id": request.get("request_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
