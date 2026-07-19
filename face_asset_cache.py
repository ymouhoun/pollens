"""Download FaceDetailer assets from a private Hugging Face repository.

The large diffusion checkpoint remains provided by RunPod Cached Models. Only
the requested face LoRA and the small, shared FaceDetailer support models are
downloaded into the worker's temporary container disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Callable


DEFAULT_CACHE_ROOT = Path("/opt/pollen/face-cache")
DEFAULT_COMFY_MODELS_ROOT = Path("/comfyui/models")
DEFAULT_FACE_ASSETS = (
    {
        "source": "assets/ultralytics/bbox/yolov11m-face.pt",
        "target": "ultralytics/bbox/yolov11m-face.pt",
    },
    {
        "source": "assets/ultralytics/segm/yolo11m-seg.pt",
        "target": "ultralytics/segm/yolo11m-seg.pt",
    },
    {
        "source": "assets/sams/sam_vit_b_01ec64.pth",
        "target": "sams/sam_vit_b_01ec64.pth",
    },
    {
        "source": "assets/upscale_models/x1_ITF_SkinDiffDetail_Lite_v1.pth",
        "target": "upscale_models/x1_ITF_SkinDiffDetail_Lite_v1.pth",
    },
)


DownloadFunction = Callable[..., str]
StatusCallback = Callable[[str, str], None]
_CACHE_LOCK = threading.Lock()


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def safe_relative_path(value: str, *, suffixes: tuple[str, ...] = ()) -> Path:
    """Return a safe POSIX relative path or raise a descriptive error."""
    raw = str(value or "").strip()
    posix_path = PurePosixPath(raw)
    if not raw or posix_path.is_absolute() or ".." in posix_path.parts:
        raise RuntimeError(f"Unsafe FaceDetailer asset path: {value!r}")
    if suffixes and posix_path.suffix.lower() not in suffixes:
        allowed = ", ".join(suffixes)
        raise RuntimeError(
            f"Unsupported FaceDetailer asset type for {value!r}; expected {allowed}"
        )
    return Path(*posix_path.parts)


def _default_downloader(**kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def configured_face_assets() -> tuple[dict[str, str], ...]:
    value = os.environ.get("POLLEN_FACE_ASSETS_JSON")
    if not value:
        return DEFAULT_FACE_ASSETS
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError("POLLEN_FACE_ASSETS_JSON is not valid JSON") from error
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("POLLEN_FACE_ASSETS_JSON must be a non-empty array")
    assets: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("source") or not item.get("target"):
            raise RuntimeError("Every FaceDetailer asset needs source and target")
        assets.append({"source": str(item["source"]), "target": str(item["target"])})
    return tuple(assets)


class FaceAssetCache:
    def __init__(
        self,
        *,
        repo_id: str,
        token: str | None,
        revision: str,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        comfy_models_root: Path = DEFAULT_COMFY_MODELS_ROOT,
        max_loras: int = 5,
        downloader: DownloadFunction = _default_downloader,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.revision = revision
        self.cache_root = Path(cache_root)
        self.comfy_models_root = Path(comfy_models_root)
        self.max_loras = max(1, max_loras)
        self.downloader = downloader

    @classmethod
    def from_environment(cls) -> "FaceAssetCache":
        repo_id = os.environ.get("HF_LORA_REPO_ID", "").strip()
        if not repo_id:
            raise RuntimeError("HF_LORA_REPO_ID is required for face detailing")
        return cls(
            repo_id=repo_id,
            token=os.environ.get("HF_TOKEN") or None,
            revision=os.environ.get("HF_LORA_REVISION", "main").strip() or "main",
            cache_root=Path(
                os.environ.get("POLLEN_FACE_CACHE_DIR", str(DEFAULT_CACHE_ROOT))
            ),
            comfy_models_root=Path(
                os.environ.get(
                    "POLLEN_COMFY_MODELS_DIR", str(DEFAULT_COMFY_MODELS_ROOT)
                )
            ),
            max_loras=_positive_int(
                os.environ.get("POLLEN_LORA_CACHE_MAX_ITEMS"), 5
            ),
        )

    def ensure(
        self,
        *,
        source: str,
        target: str,
        pinned: bool,
        status_callback: StatusCallback | None = None,
    ) -> Path:
        source_path = safe_relative_path(
            source, suffixes=(".safetensors", ".pt", ".pth")
        )
        target_path = safe_relative_path(
            target, suffixes=(".safetensors", ".pt", ".pth")
        )
        cache_key = hashlib.sha256(
            f"{self.repo_id}\0{self.revision}\0{source_path.as_posix()}".encode()
        ).hexdigest()[:24]
        item_root = self.cache_root / ("pinned" if pinned else "loras") / cache_key
        target_file = self.comfy_models_root / target_path

        with _CACHE_LOCK:
            if target_file.is_symlink():
                try:
                    resolved_target = target_file.resolve()
                    if (
                        resolved_target.is_file()
                        and item_root.resolve() in resolved_target.parents
                    ):
                        if status_callback:
                            status_callback("cached", source_path.as_posix())
                        item_root.touch()
                        if not pinned:
                            self._prune_loras(exclude=cache_key)
                        return target_file
                except FileNotFoundError:
                    pass
            item_root.mkdir(parents=True, exist_ok=True)
            if status_callback:
                status_callback("downloading", source_path.as_posix())
            downloaded = Path(
                self.downloader(
                    repo_id=self.repo_id,
                    filename=source_path.as_posix(),
                    revision=self.revision,
                    token=self.token,
                    cache_dir=str(item_root / "huggingface"),
                    local_files_only=False,
                )
            )
            if not downloaded.is_file():
                raise RuntimeError(
                    f"Hugging Face did not return a file for {source_path.as_posix()}"
                )

            target_file.parent.mkdir(parents=True, exist_ok=True)
            if target_file.exists() or target_file.is_symlink():
                try:
                    if target_file.resolve() == downloaded.resolve():
                        item_root.touch()
                        if not pinned:
                            self._prune_loras(exclude=cache_key)
                        return target_file
                except FileNotFoundError:
                    pass
                if target_file.is_symlink():
                    target_file.unlink()
                else:
                    raise RuntimeError(
                        f"FaceDetailer target already exists: {target_file}"
                    )

            os.symlink(downloaded, target_file)
            (item_root / "pollen-cache.json").write_text(
                json.dumps(
                    {
                        "source": source_path.as_posix(),
                        "target": target_path.as_posix(),
                        "pinned": pinned,
                    }
                ),
                encoding="utf-8",
            )
            item_root.touch()
            if not pinned:
                self._prune_loras(exclude=cache_key)
            return target_file

    def _prune_loras(self, *, exclude: str) -> None:
        lora_root = self.cache_root / "loras"
        if not lora_root.is_dir():
            return
        candidates = sorted(
            (path for path in lora_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        keep = {path.name for path in candidates[: self.max_loras]}
        keep.add(exclude)
        for item_root in candidates:
            if item_root.name in keep:
                continue
            metadata_path = item_root / "pollen-cache.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                target = self.comfy_models_root / safe_relative_path(
                    metadata["target"]
                )
                if target.is_symlink():
                    target.unlink()
            except (FileNotFoundError, KeyError, json.JSONDecodeError, RuntimeError):
                pass
            shutil.rmtree(item_root, ignore_errors=True)


def ensure_face_dependencies(
    job: dict[str, Any],
    status_callback: StatusCallback | None = None,
) -> None:
    """Prepare the selected LoRA and shared models before ComfyUI validates the job."""
    job_input = job.get("input", {})
    if not isinstance(job_input, dict):
        return
    face_lora = job_input.get("faceLora")
    if not face_lora:
        return

    source_path = safe_relative_path(
        str(face_lora), suffixes=(".safetensors",)
    )
    if not source_path.parts or source_path.parts[0] != "loras":
        raise RuntimeError("faceLora must point inside the loras/ directory")
    if len(source_path.parts) < 2:
        raise RuntimeError("faceLora must include a LoRA filename")

    cache = FaceAssetCache.from_environment()
    for asset in configured_face_assets():
        cache.ensure(
            source=asset["source"],
            target=asset["target"],
            pinned=True,
            status_callback=status_callback,
        )

    comfy_lora_path = Path("loras", *source_path.parts[1:])
    cache.ensure(
        source=source_path.as_posix(),
        target=comfy_lora_path.as_posix(),
        pinned=False,
        status_callback=status_callback,
    )
