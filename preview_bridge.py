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
        self.last_sent_at = float("-inf")
        self.interval_seconds = _positive_int(
            os.environ.get("POLLEN_PREVIEW_INTERVAL_MS"), 750
        ) / 1000
        self.max_bytes = _positive_int(
            os.environ.get("POLLEN_PREVIEW_MAX_BYTES"), 500_000
        )

    def observe(self, message: str | bytes) -> None:
        if isinstance(message, str):
            progress = extract_progress(message)
            if progress:
                self.step, self.total = progress
            return

        if not isinstance(message, (bytes, bytearray)):
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
        if now - self.last_sent_at < self.interval_seconds:
            return

        encoded = base64.b64encode(image).decode("ascii")
        self.send_update(
            self.job,
            {
                "step": self.step,
                "total": self.total,
                "previewImage": f"data:{mime_type};base64,{encoded}",
            },
        )
        self.last_sent_at = now
        print(
            "pollen-preview - Published preview "
            f"step={self.step}/{self.total} bytes={len(image)}"
        )
