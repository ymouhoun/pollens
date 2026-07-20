"""Persistent, isolated GPU worker used by the ComfyUI process."""

from __future__ import annotations

import atexit
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid
from typing import Any


class GPUWorkerError(RuntimeError):
    pass


class GPUWorkerCrashed(GPUWorkerError):
    pass


class GPUWorkerTimeout(GPUWorkerError):
    pass


class GPUWorkerClient:
    """Serialize requests to one model-caching worker process."""

    def __init__(
        self,
        worker_script: str | Path | None = None,
        startup_timeout: float = 30.0,
        request_timeout: float | None = None,
    ) -> None:
        self.worker_script = Path(worker_script or Path(__file__).with_name("gpu_worker.py"))
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout or float(
            os.environ.get("LLM_ENHANCER_GPU_TIMEOUT_SECONDS", "900")
        )
        self._process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._request_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._worker_info: dict[str, Any] = {}

    @property
    def worker_info(self) -> dict[str, Any]:
        return dict(self._worker_info)

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self._alive():
                return self.worker_info

            if self._process is not None:
                self._terminate_locked(force=True)
            message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
            stderr_tail: deque[str] = deque(maxlen=40)
            self._messages = message_queue
            self._stderr_tail = stderr_tail
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            worker_python = env.get("LLM_ENHANCER_WORKER_PYTHON", sys.executable)
            process = subprocess.Popen(
                [worker_python, "-u", str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
            self._process = process
            threading.Thread(
                target=self._read_stdout,
                args=(process, message_queue, stderr_tail),
                name="llm-enhancer-gpu-stdout",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._read_stderr,
                args=(process, stderr_tail),
                name="llm-enhancer-gpu-stderr",
                daemon=True,
            ).start()

            try:
                message = self._next_message(self.startup_timeout)
            except BaseException:
                self._terminate_locked(force=True)
                raise
            if message.get("type") == "ready":
                self._worker_info = message
                print(
                    "[LLM Enhancer] GPU worker ready: "
                    f"{message.get('gpu_name')} (sm_{message.get('sm')}) / "
                    f"llama-cpp-python {message.get('llama_cpp_version')}"
                    + (
                        ""
                        if message.get("backend_version_certified", True)
                        else " (version non certifiée, validation par génération réelle)"
                    )
                )
                return self.worker_info

            self._terminate_locked()
            detail = message.get("error") or self._diagnostic_tail()
            raise GPUWorkerError(f"GPU worker failed to start: {detail}")

    def _read_stdout(
        self,
        process: subprocess.Popen[str],
        message_queue: queue.Queue[dict[str, Any]],
        stderr_tail: deque[str],
    ) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message_queue.put(json.loads(line))
            except json.JSONDecodeError:
                message_queue.put(
                    {"type": "protocol_error", "error": f"Invalid worker output: {line[:500]}"}
                )
        message_queue.put(
            {
                "type": "eof",
                "returncode": process.poll(),
                "error": "\n".join(stderr_tail) or "worker exited without diagnostics",
            }
        )

    def _read_stderr(
        self,
        process: subprocess.Popen[str],
        stderr_tail: deque[str],
    ) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            line = line.rstrip()
            if line:
                stderr_tail.append(line)
                print(f"[LLM Enhancer GPU] {line}")

    def _diagnostic_tail(self) -> str:
        return "\n".join(self._stderr_tail) or "worker exited without diagnostics"

    def _next_message(self, timeout: float) -> dict[str, Any]:
        try:
            return self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise GPUWorkerTimeout(f"GPU worker did not respond within {timeout:.0f}s") from exc

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._request_lock:
            self.start()
            request_id = uuid.uuid4().hex
            request = dict(payload)
            request["type"] = "generate"
            request["request_id"] = request_id
            process = self._process
            assert process is not None and process.stdin is not None

            try:
                process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise GPUWorkerCrashed(
                    f"GPU worker stopped before generation: {self._diagnostic_tail()}"
                ) from exc

            deadline = time.monotonic() + self.request_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.stop(force=True)
                    raise GPUWorkerTimeout(
                        f"GPU generation exceeded {self.request_timeout:.0f}s; worker stopped"
                    )
                message = self._next_message(remaining)
                message_type = message.get("type")
                if message_type in {"eof", "protocol_error", "fatal"}:
                    raise GPUWorkerCrashed(
                        message.get("error") or f"GPU worker exited ({message.get('returncode')})"
                    )
                if message.get("request_id") != request_id:
                    continue
                if message_type == "error":
                    raise GPUWorkerError(message.get("error", "unknown GPU inference error"))
                if message_type != "result":
                    raise GPUWorkerError(f"Unexpected GPU worker response: {message_type}")
                return message

    def _terminate_locked(self, force: bool = False) -> None:
        process = self._process
        self._process = None
        self._worker_info = {}
        if process is None:
            return
        try:
            if process.poll() is not None:
                return
            if not force and process.stdin is not None:
                try:
                    process.stdin.write('{"type":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=3)
                    return
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    pass
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def stop(self, force: bool = False) -> None:
        with self._lifecycle_lock:
            self._terminate_locked(force=force)


_DEFAULT_WORKER = GPUWorkerClient()
atexit.register(_DEFAULT_WORKER.stop)


def generate_on_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    return _DEFAULT_WORKER.request(payload)


def stop_gpu_worker() -> None:
    _DEFAULT_WORKER.stop()
