"""Run the official worker-comfyui handler with progressive previews enabled."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from typing import Any

import runpod

from face_asset_cache import ensure_face_dependencies
from preview_bridge import PreviewBridge


UPSTREAM_HANDLER_PATH = os.environ.get(
    "POLLEN_UPSTREAM_HANDLER_PATH", "/handler.py"
)


def _load_upstream_handler():
    # The official handler imports /network_volume.py from the image root.
    if "/" not in sys.path:
        sys.path.insert(0, "/")
    spec = importlib.util.spec_from_file_location(
        "pollen_upstream_worker_handler", UPSTREAM_HANDLER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {UPSTREAM_HANDLER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upstream = _load_upstream_handler()
_original_recv = upstream.websocket.WebSocket.recv
_context = threading.local()


def _send_progress(job: dict[str, Any], payload: dict[str, Any]) -> None:
    updater = getattr(runpod.serverless, "progress_update", None)
    if updater is None:
        # Compatibility with the SDK version pinned by worker-comfyui 5.8.6.
        from runpod.serverless.modules.rp_progress import progress_update

        updater = progress_update
    updater(job, payload)


def _recv_with_preview(socket, *args, **kwargs):
    message = _original_recv(socket, *args, **kwargs)
    bridge = getattr(_context, "preview_bridge", None)
    if bridge is not None:
        try:
            bridge.observe(message)
        except Exception as error:  # A preview must never fail the generation.
            print(f"pollen-preview - Non-fatal preview error: {error}")
    return message


# The official handler keeps receiving every message and remains responsible
# for completion, reconnection, history retrieval and final image encoding.
upstream.websocket.WebSocket.recv = _recv_with_preview


def handler(job: dict[str, Any]):
    ensure_face_dependencies(job)
    enabled = os.environ.get("POLLEN_PREVIEW_ENABLED", "true").lower() == "true"
    if enabled:
        _context.preview_bridge = PreviewBridge(job, _send_progress)
        print("pollen-preview - Progressive previews enabled")
    try:
        return upstream.handler(job)
    finally:
        if hasattr(_context, "preview_bridge"):
            del _context.preview_bridge


if __name__ == "__main__":
    print("pollen-preview - Starting wrapped worker-comfyui handler")
    runpod.serverless.start({"handler": handler})
