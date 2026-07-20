"""Translate ComfyUI websocket previews into RunPod progress payloads.

The upstream worker-comfyui handler intentionally ignores binary websocket
messages. ComfyUI uses those messages for latent previews. This module keeps
the protocol-specific code isolated so the upstream handler can stay intact.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import time
from collections.abc import Callable
from typing import Any


PREVIEW_IMAGE = 1
PREVIEW_IMAGE_WITH_METADATA = 4


STAGE_LABELS = {
    "preparing_worker": "Preparing worker",
    "preparing_face_assets": "Preparing Face Detail assets",
    "starting_comfy": "Starting ComfyUI",
    "comfy_ready": "ComfyUI ready",
    "uploading_source": "Uploading source image",
    "queueing_workflow": "Sending workflow to ComfyUI",
    "workflow_queued": "Workflow queued",
    "enhancing_prompt": "Enhancing prompt · Qwen 8B",
    "loading_models": "Loading models",
    "detecting_face": "Detecting face",
    "refining_face": "Refining face",
    "sampling": "Generating image",
    "decoding": "Decoding image",
    "refining_details": "Refining details",
    "saving": "Saving image",
    "finalizing": "Finalizing",
    "executing": "Executing workflow",
}


def _workflow_node(job: dict[str, Any], node_id: Any) -> dict[str, Any] | None:
    workflow = job.get("input", {}).get("workflow", {})
    if not isinstance(workflow, dict):
        return None
    node = workflow.get(str(node_id))
    return node if isinstance(node, dict) else None


def stage_for_node(job: dict[str, Any], node_id: Any) -> tuple[str, str, str]:
    """Translate the currently executing ComfyUI node into a studio stage."""
    node = _workflow_node(job, node_id) or {}
    class_type = str(node.get("class_type") or "")
    metadata = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
    title = str(metadata.get("title") or class_type or f"Node {node_id}")
    searchable = f"{class_type} {title}".lower()

    if "llmpromptenhancer" in searchable or "prompt enhancer" in searchable:
        stage = "enhancing_prompt"
    elif any(
        token in searchable
        for token in (
            "loader",
            "load diffusion",
            "load clip",
            "load vae",
            "load lora",
            "power lora",
        )
    ):
        stage = "loading_models"
    elif "pollenfacedetailer" in searchable or "facedetailer" in searchable:
        stage = "refining_face"
    elif any(token in searchable for token in ("ultralytics", "detector", "samloader")):
        stage = "detecting_face"
    elif any(token in searchable for token in ("ksampler", "samplercustom", "sampler")):
        stage = "sampling"
    elif any(token in searchable for token in ("vaedecode", "vae decode", "decode")):
        stage = "decoding"
    elif any(token in searchable for token in ("upscale", "imagescale")):
        stage = "refining_details"
    elif any(token in searchable for token in ("saveimage", "save image", "output")):
        stage = "saving"
    else:
        stage = "executing"

    return stage, STAGE_LABELS[stage], title


def _first_history_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                return item
    return None


def _history_output_maps(value: Any):
    if not isinstance(value, dict):
        return
    outputs = value.get("outputs")
    if isinstance(outputs, dict):
        yield outputs
    for nested in value.values():
        if isinstance(nested, dict):
            yield from _history_output_maps(nested)


def extract_prompt_enhancer_metadata(
    job: dict[str, Any], history: Any
) -> dict[str, str]:
    """Extract the enhancer's actual text outputs from ComfyUI history."""
    workflow = job.get("input", {}).get("workflow", {})
    if not isinstance(workflow, dict):
        return {}

    enhancer_nodes = {
        str(node_id): node
        for node_id, node in workflow.items()
        if isinstance(node, dict)
        and str(node.get("class_type") or "").lower() == "llmpromptenhancer"
    }
    if not enhancer_nodes:
        return {}

    metadata: dict[str, str] = {}
    node_id, workflow_node = next(iter(enhancer_nodes.items()))
    inputs = workflow_node.get("inputs", {})
    if isinstance(inputs, dict) and isinstance(inputs.get("style_preset"), str):
        metadata["enhancerPreset"] = inputs["style_preset"]

    for outputs in _history_output_maps(history):
        node_output = outputs.get(node_id)
        if not isinstance(node_output, dict):
            continue
        enhanced = _first_history_text(node_output.get("text"))
        negative = _first_history_text(node_output.get("negative_prompt"))
        preset = _first_history_text(node_output.get("style_preset"))
        if enhanced:
            metadata["enhancedPrompt"] = enhanced
        if negative:
            metadata["enhancedNegativePrompt"] = negative
        if preset:
            metadata["enhancerPreset"] = preset
        break

    return metadata


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def infer_workflow_steps(job: dict[str, Any]) -> int:
    """Return the largest numeric ``steps`` input found in the workflow."""
    workflow = job.get("input", {}).get("workflow", {})
    if not isinstance(workflow, dict):
        return 0

    totals: list[int] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        value = inputs.get("steps")
        if isinstance(value, (int, float)) and value > 0:
            totals.append(int(value))
    return max(totals, default=0)


def _mime_from_bytes(image: bytes, fallback: str = "image/jpeg") -> str:
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return fallback


def decode_binary_preview(message: bytes) -> tuple[str, bytes] | None:
    """Decode legacy and metadata-aware ComfyUI preview frames.

    Legacy layout::

        uint32 event=1 | uint32 image_type | encoded image bytes

    Metadata layout::

        uint32 event=4 | uint32 metadata_length | metadata JSON | image bytes
    """
    if len(message) < 8:
        return None

    event = struct.unpack(">I", message[:4])[0]
    if event == PREVIEW_IMAGE:
        image_type = struct.unpack(">I", message[4:8])[0]
        image = message[8:]
        fallback = "image/png" if image_type == 2 else "image/jpeg"
        return (_mime_from_bytes(image, fallback), image) if image else None

    if event == PREVIEW_IMAGE_WITH_METADATA:
        metadata_length = struct.unpack(">I", message[4:8])[0]
        image_offset = 8 + metadata_length
        if image_offset > len(message):
            return None

        metadata: dict[str, Any] = {}
        try:
            metadata = json.loads(message[8:image_offset].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

        image = message[image_offset:]
        if not image:
            return None
        fallback = str(metadata.get("image_type") or "image/jpeg")
        return _mime_from_bytes(image, fallback), image

    return None


def extract_progress(message: str) -> tuple[int, int] | None:
    """Extract the active sampler progress from ComfyUI JSON messages."""
    try:
        payload = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return None

    data = payload.get("data", {})
    if payload.get("type") == "progress":
        try:
            return int(data.get("value", 0)), int(data.get("max", 0))
        except (TypeError, ValueError):
            return None

    if payload.get("type") != "progress_state":
        return None

    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        return None

    candidates = [
        node
        for node in nodes.values()
        if isinstance(node, dict)
        and node.get("state") in {"running", "in_progress"}
    ]
    if not candidates:
        candidates = [node for node in nodes.values() if isinstance(node, dict)]
    if not candidates:
        return None

    node = candidates[-1]
    try:
        return int(node.get("value", 0)), int(node.get("max", 0))
    except (TypeError, ValueError):
        return None


class PreviewBridge:
    """Observe ComfyUI websocket messages and publish throttled previews."""

    def __init__(
        self,
        job: dict[str, Any],
        send_update: Callable[[dict[str, Any], dict[str, Any]], None],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.job = job
        self.send_update = send_update
        self.clock = clock
        self.step = 0
        self.total = infer_workflow_steps(job)
        self.stage = "preparing_worker"
        self.stage_label = STAGE_LABELS[self.stage]
        self.detail: str | None = None
        self.current_node_id: str | None = None
        self.last_preview_sent_at = float("-inf")
        self.last_status_sent_at = float("-inf")
        self.last_status_signature: tuple[Any, ...] | None = None
        self.interval_seconds = _positive_int(
            os.environ.get("POLLEN_PREVIEW_INTERVAL_MS"), 750
        ) / 1000
        self.status_interval_seconds = _positive_int(
            os.environ.get("POLLEN_STATUS_INTERVAL_MS"), 500
        ) / 1000
        self.max_bytes = _positive_int(
            os.environ.get("POLLEN_PREVIEW_MAX_BYTES"), 500_000
        )
        self.previews_enabled = (
            os.environ.get("POLLEN_PREVIEW_ENABLED", "true").lower() == "true"
        )

    def _payload(self, **extra: Any) -> dict[str, Any]:
        payload = {
            "stage": self.stage,
            "stageLabel": self.stage_label,
            "detail": self.detail,
            "step": self.step,
            "total": self.total,
        }
        payload.update(extra)
        return payload

    def publish_stage(
        self,
        stage: str,
        label: str | None = None,
        detail: str | None = None,
        *,
        force: bool = True,
    ) -> None:
        previous = (self.stage, self.stage_label, self.detail)
        self.stage = stage
        self.stage_label = label or STAGE_LABELS.get(stage, stage.replace("_", " ").title())
        self.detail = detail
        signature = (self.stage, self.stage_label, self.detail, self.step, self.total)
        now = self.clock()
        stage_changed = previous != (self.stage, self.stage_label, self.detail)
        if not force and not stage_changed:
            if now - self.last_status_sent_at < self.status_interval_seconds:
                return
            if signature == self.last_status_signature:
                return
        self.send_update(self.job, self._payload())
        self.last_status_sent_at = now
        self.last_status_signature = signature
        print(
            "pollen-status - Published stage "
            f"{self.stage} ({self.detail or self.stage_label})"
        )

    def _observe_json(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return

        event_type = payload.get("type")
        data = payload.get("data", {})
        if not isinstance(data, dict):
            data = {}

        if event_type == "execution_start":
            self.publish_stage("workflow_queued", STAGE_LABELS["workflow_queued"])
            return

        if event_type == "executing":
            node_id = data.get("node")
            if node_id is None:
                self.publish_stage("finalizing", STAGE_LABELS["finalizing"])
                return
            self.current_node_id = str(node_id)
            stage, label, detail = stage_for_node(self.job, node_id)
            self.publish_stage(stage, label, detail)
            return

        if event_type == "execution_success":
            self.publish_stage("finalizing", STAGE_LABELS["finalizing"])
            return

        progress = extract_progress(message)
        if not progress:
            return
        self.step, self.total = progress
        if self.stage != "refining_face":
            self.stage = "sampling"
            self.stage_label = STAGE_LABELS[self.stage]
            node = _workflow_node(self.job, self.current_node_id)
            metadata = node.get("_meta") if isinstance(node, dict) else None
            self.detail = (
                str(metadata.get("title"))
                if isinstance(metadata, dict) and metadata.get("title")
                else self.detail
            )
        self.publish_stage(
            self.stage,
            self.stage_label,
            self.detail,
            force=False,
        )

    def observe(self, message: str | bytes) -> None:
        if isinstance(message, str):
            self._observe_json(message)
            return

        if not isinstance(message, (bytes, bytearray)):
            return

        if not self.previews_enabled:
            return

        decoded = decode_binary_preview(bytes(message))
        if not decoded:
            return

        mime_type, image = decoded
        if len(image) > self.max_bytes:
            print(
                "pollen-preview - Skipping oversized preview "
                f"({len(image)} bytes > {self.max_bytes})"
            )
            return

        now = self.clock()
        if now - self.last_preview_sent_at < self.interval_seconds:
            return

        encoded = base64.b64encode(image).decode("ascii")
        self.send_update(
            self.job,
            self._payload(previewImage=f"data:{mime_type};base64,{encoded}"),
        )
        self.last_preview_sent_at = now
        print(
            "pollen-preview - Published preview "
            f"step={self.step}/{self.total} bytes={len(image)}"
        )
