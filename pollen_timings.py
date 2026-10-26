"""Where a job's time goes, measured on the worker.

Returned with every result as ``output.pollenTimings`` and stored by the app
(InferenceRun), so the admin can split the wait between GPU search, worker
start, model loading and generation. Measuring must never fail a job: every
public function here swallows its own errors.

Clock: wall-clock epoch milliseconds (``time.time()``), so the app can line the
worker's marks up with the moment the job was sent.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any


BOOT_FILE = os.environ.get("POLLEN_BOOT_TIMINGS", "/tmp/pollen-boot.json")
MAX_NODES = 60

# Process-wide facts: one worker process serves many jobs
_PROCESS: dict[str, Any] = {
    "handler_started_at": time.time(),
    "comfy_ready_at": None,
    "jobs_served": 0,
}


def _ms(value: float | None) -> int | None:
    return None if value is None else int(round(value * 1000))


def container_started_at(proc_root: str = "/proc") -> float | None:
    """When the container started: the start of PID 1, read from /proc.

    /proc/1/stat gives the start in clock ticks since the host booted, and
    /proc/stat gives the host's boot time. bootstrap.py is PID 1 (CMD) and
    execs /start.sh, so PID 1 is the container's own first process.
    """
    try:
        with open(f"{proc_root}/stat", encoding="utf-8") as handle:
            btime = next(
                int(line.split()[1]) for line in handle if line.startswith("btime ")
            )
        with open(f"{proc_root}/1/stat", encoding="utf-8") as handle:
            stat = handle.read()
        # Fields after "(comm)": index 0 is field 3 (state), so starttime (field 22) is index 19
        fields = stat[stat.rindex(")") + 2 :].split()
        ticks = int(fields[19])
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


_GPU_NAME: list[str | None] = []


def gpu_name() -> str | None:
    """The GPU this worker runs on, read once from nvidia-smi (the app's RunPod lookup sometimes has none)."""
    if _GPU_NAME:
        return _GPU_NAME[0]
    name = None
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
        ).stdout.strip().splitlines()
        name = out[0].strip() if out and out[0].strip() else None
    except Exception:
        name = None
    _GPU_NAME.append(name)
    return name


def write_boot_file(data: dict[str, Any], path: str | None = None) -> None:
    """Called by bootstrap.py: when it started, when the model links were ready, which revision."""
    try:
        with open(path or BOOT_FILE, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
    except Exception as error:  # informative only
        print(f"pollen-timings - boot file not written: {error}", flush=True)


def read_boot_file(path: str | None = None) -> dict[str, Any]:
    try:
        with open(path or BOOT_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def note_comfy_ready(at: float | None = None) -> None:
    """The first time ComfyUI answered in this process."""
    if _PROCESS["comfy_ready_at"] is None:
        _PROCESS["comfy_ready_at"] = at if at is not None else time.time()


def node_category(class_type: str, title: str = "") -> str:
    """What a ComfyUI node spends its time on, for the admin's split."""
    kind = class_type.lower()
    text = f"{kind} {title.lower()}"
    if "llmpromptenhancer" in kind or "prompt enhancer" in text:
        return "enhance"
    if "facedetailer" in kind:
        return "face"
    if "loader" in kind or "detectorprovider" in kind or "load lora" in text or "power lora" in text:
        return "load"
    if "textencode" in kind or "cliptextencode" in kind:
        return "encode"
    if "sampler" in kind:
        return "sample"
    if "upscale" in kind:
        return "upscale"
    if "vaedecode" in kind or "decode" in kind:
        return "decode"
    if "saveimage" in kind or "previewimage" in kind:
        return "save"
    return "other"


PHASES = (
    "faceAssets", "comfyWait", "upload", "queue",
    "load", "encode", "enhance", "vram", "sample", "face", "upscale", "decode", "save", "other",
    "output",
)


class JobTimer:
    """Marks of one job: handler spans, then one span per ComfyUI node."""

    def __init__(
        self,
        job: dict[str, Any],
        *,
        clock: Callable[[], float] = time.time,
        boot: dict[str, Any] | None = None,
        container_start: float | None = None,
    ) -> None:
        self.clock = clock
        self.job = job
        self.received_at = clock()
        self.job_index = _PROCESS["jobs_served"]
        _PROCESS["jobs_served"] += 1
        self.boot = boot if boot is not None else read_boot_file()
        self.container_start = container_start if container_start is not None else container_started_at()
        self.phases: dict[str, float] = {}
        self.nodes: list[dict[str, Any]] = []
        self.cached: list[str] = []
        self.downloads: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.execution_start: float | None = None
        self.execution_end: float | None = None
        self.finished_at: float | None = None
        self.steps = 0

    # ---- handler spans -------------------------------------------------
    def add(self, phase: str, seconds: float) -> None:
        if seconds > 0:
            self.phases[phase] = self.phases.get(phase, 0.0) + seconds

    def timed(self, phase: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        start = self.clock()
        try:
            return function(*args, **kwargs)
        finally:
            self.add(phase, self.clock() - start)

    def download(self, filename: str, seconds: float) -> None:
        self.downloads.append({"file": filename, "ms": _ms(seconds)})

    # ---- ComfyUI websocket ---------------------------------------------
    def _workflow_node(self, node_id: str) -> dict[str, Any]:
        workflow = self.job.get("input", {}).get("workflow", {})
        node = workflow.get(node_id) if isinstance(workflow, dict) else None
        return node if isinstance(node, dict) else {}

    def _close_current(self, at: float) -> None:
        node = self.current
        if node is None:
            return
        self.current = None
        total = max(0.0, at - node["start"])
        category = node["category"]
        if category == "sample" and node.get("first_step") is not None:
            # From the sampler's start to its first step: ComfyUI moves the model into VRAM (and conditioning)
            prep = max(0.0, node["first_step"] - node["start"])
            self.add("vram", prep)
            self.add("sample", total - prep)
        else:
            self.add(category, total)
        if len(self.nodes) < MAX_NODES:
            entry = {
                "id": node["id"],
                "class": node["class"],
                "title": node["title"],
                "category": category,
                "ms": _ms(total),
            }
            if node.get("first_step") is not None:
                entry["toFirstStepMs"] = _ms(node["first_step"] - node["start"])
            self.nodes.append(entry)

    def observe(self, message: Any) -> None:
        if not isinstance(message, str):
            return
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        kind = payload.get("type")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        now = self.clock()

        if kind == "execution_start":
            if self.execution_start is None:
                self.execution_start = now
        elif kind == "execution_cached":
            nodes = data.get("nodes")
            if isinstance(nodes, list):
                self.cached.extend(str(n) for n in nodes)
        elif kind == "executing":
            node_id = data.get("node")
            self._close_current(now)
            if node_id is None:
                self.execution_end = now
                return
            if self.execution_start is None:
                self.execution_start = now
            node = self._workflow_node(str(node_id))
            class_type = str(node.get("class_type") or "")
            meta = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
            title = str(meta.get("title") or class_type)
            self.current = {
                "id": str(node_id),
                "class": class_type,
                "title": title,
                "category": node_category(class_type, title),
                "start": now,
                "first_step": None,
            }
        elif kind in ("progress", "progress_state"):
            if self.current is not None and self.current.get("first_step") is None:
                self.current["first_step"] = now
            if kind == "progress":
                try:
                    self.steps = max(self.steps, int(data.get("max", 0)))
                except (TypeError, ValueError):
                    pass
        elif kind in ("execution_success", "execution_error", "execution_interrupted"):
            self._close_current(now)
            if self.execution_end is None:
                self.execution_end = now

    # ---- result --------------------------------------------------------
    def finish(self) -> dict[str, Any]:
        self.finished_at = self.clock()
        self._close_current(self.execution_end or self.finished_at)
        if self.execution_end is not None:
            self.add("output", self.finished_at - self.execution_end)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        boot = self.boot or {}
        return {
            "version": 1,
            "worker": {
                "containerStartedAt": _ms(self.container_start),
                "bootstrapStartedAt": _ms(boot.get("bootstrapStartedAt")),
                "linksReadyAt": _ms(boot.get("linksReadyAt")),
                "handlerStartedAt": _ms(_PROCESS["handler_started_at"]),
                "comfyReadyAt": _ms(_PROCESS["comfy_ready_at"]),
                "jobIndex": self.job_index,
                "modelId": boot.get("modelId"),
                "revision": boot.get("revision"),
                "checkpoints": boot.get("checkpoints") or [],
                "gpu": gpu_name(),
            },
            "job": {
                "receivedAt": _ms(self.received_at),
                "executionStartAt": _ms(self.execution_start),
                "executionEndAt": _ms(self.execution_end),
                "finishedAt": _ms(self.finished_at),
                "warmup": bool(self.job.get("input", {}).get("warmup")),
            },
            "phases": {name: _ms(self.phases[name]) for name in PHASES if name in self.phases},
            "nodes": self.nodes,
            "cached": self.cached[:MAX_NODES],
            "downloads": self.downloads,
            "steps": self.steps,
        }
