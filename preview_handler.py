"""Run the official worker-comfyui handler with progressive previews enabled."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from typing import Any

import runpod

from face_asset_cache import ensure_face_dependencies
from pollen_timings import JobTimer, note_comfy_ready
from preview_bridge import PreviewBridge, extract_prompt_enhancer_metadata


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
    timer = getattr(_context, "timer", None)
    if timer is not None:
        try:
            timer.observe(message)
        except Exception as error:  # Measuring must never fail the generation.
            print(f"pollen-timings - Non-fatal timing error: {error}")
    bridge = getattr(_context, "preview_bridge", None)
    if bridge is not None:
        try:
            bridge.observe(message)
        except Exception as error:  # A preview must never fail the generation.
            print(f"pollen-preview - Non-fatal preview error: {error}")
    return message


def _publish_stage(stage: str, label: str, detail: str | None = None) -> None:
    bridge = getattr(_context, "preview_bridge", None)
    if bridge is not None:
        bridge.publish_stage(stage, label, detail)


def _wrap_upstream_function(
    name: str,
    before: tuple[str, str, str | None],
    after: tuple[str, str, str | None] | None = None,
    phase: str | None = None,
) -> None:
    original = getattr(upstream, name, None)
    if original is None:
        return

    def wrapped(*args, **kwargs):
        _publish_stage(*before)
        timer = getattr(_context, "timer", None)
        result = (
            timer.timed(phase, original, *args, **kwargs)
            if timer is not None and phase
            else original(*args, **kwargs)
        )
        if name == "check_server" and result:
            note_comfy_ready()
        if after is not None and (name != "check_server" or result):
            _publish_stage(*after)
        return result

    setattr(upstream, name, wrapped)


# The official handler keeps receiving every message and remains responsible
# for completion, reconnection, history retrieval and final image encoding.
upstream.websocket.WebSocket.recv = _recv_with_preview
_wrap_upstream_function(
    "check_server",
    ("starting_comfy", "Starting ComfyUI", None),
    ("comfy_ready", "ComfyUI ready", None),
    "comfyWait",
)
_wrap_upstream_function(
    "upload_images",
    ("uploading_source", "Uploading source image", None),
    phase="upload",
)
_wrap_upstream_function(
    "queue_workflow",
    ("queueing_workflow", "Sending workflow to ComfyUI", None),
    ("workflow_queued", "Workflow queued", None),
    "queue",
)
_wrap_upstream_function(
    "get_image_data",
    ("finalizing", "Downloading final image", None),
)


def _wrap_history_capture() -> None:
    original = getattr(upstream, "get_history", None)
    if original is None:
        return

    def wrapped(*args, **kwargs):
        _publish_stage("finalizing", "Collecting output")
        history = original(*args, **kwargs)
        _context.prompt_history = history
        return history

    upstream.get_history = wrapped


_wrap_history_capture()


def _start_timer(job: dict[str, Any]):
    try:
        return JobTimer(job)
    except Exception as error:  # Measuring must never fail the generation.
        print(f"pollen-timings - Timer not started: {error}")
        return None


def handler(job: dict[str, Any]):
    _context.timer = _start_timer(job)
    _context.preview_bridge = PreviewBridge(job, _send_progress)
    _context.prompt_history = None
    bridge = _context.preview_bridge
    timer = _context.timer
    download = {"file": None, "start": 0.0}

    def close_download() -> None:
        if timer is not None and download["file"]:
            timer.download(download["file"], timer.clock() - download["start"])
        download["file"] = None

    try:
        bridge.publish_stage("preparing_worker", "Preparing worker")

        def face_asset_status(state: str, source: str) -> None:
            close_download()
            if state == "downloading" and timer is not None:
                download["file"] = os.path.basename(source)
                download["start"] = timer.clock()
            filename = os.path.basename(source)
            detail = (
                f"Downloading {filename}"
                if state == "downloading"
                else f"Using cached {filename}"
            )
            bridge.publish_stage(
                "preparing_face_assets",
                "Preparing Face Detail assets",
                detail,
            )

        if timer is not None:
            timer.timed(
                "faceAssets",
                ensure_face_dependencies,
                job,
                status_callback=face_asset_status,
            )
        else:
            ensure_face_dependencies(job, status_callback=face_asset_status)
        close_download()
        print("pollen-preview - Structured ComfyUI progress enabled")
        result = upstream.handler(job)
        if isinstance(result, dict):
            result.update(
                extract_prompt_enhancer_metadata(
                    job,
                    getattr(_context, "prompt_history", None),
                )
            )
            if timer is not None:
                try:
                    result["pollenTimings"] = timer.finish()
                except Exception as error:  # informative only
                    print(f"pollen-timings - Summary failed: {error}")
        return result
    finally:
        if hasattr(_context, "timer"):
            del _context.timer
        if hasattr(_context, "preview_bridge"):
            del _context.preview_bridge
        if hasattr(_context, "prompt_history"):
            del _context.prompt_history


if __name__ == "__main__":
    print("pollen-preview - Starting wrapped worker-comfyui handler")
    runpod.serverless.start({"handler": handler})
