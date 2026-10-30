"""Input images given by address instead of base64.

The Pollens functions hand the worker short-lived signed addresses of the libraries' images
(beta.pollens.ai/media/…?e=…&s=…) rather than the images themselves: a full-resolution PNG, once in base64,
does not fit in a RunPod request (10 MB), and a JPEG would damage the fine-tune's grain.

Each entry of job["input"]["images"] may carry {"name", "url"} instead of {"name", "image"}: the file is
downloaded here and handed to the official handler as base64, which it then uploads to ComfyUI as before.
Entries that already carry "image" are left untouched (the face detail and the older calls).
"""

from __future__ import annotations

import base64
import urllib.request
from typing import Any, Callable
from urllib.parse import urlparse

MAX_BYTES = 60 * 1024 * 1024
TIMEOUT_S = 60
IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")


class RemoteImageError(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A signed address answers with the file itself: a redirect is refused, so it cannot lead anywhere else."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "pollens-worker"})
    with _opener.open(request, timeout=TIMEOUT_S) as response:  # https checked by the caller
        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        data = response.read(MAX_BYTES + 1)
    return data, content_type


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def inline_remote_images(
    job: dict[str, Any],
    fetch: Callable[[str], tuple[bytes, str]] = _fetch,
    status: Callable[[str], None] | None = None,
) -> int:
    """Replace every {"name", "url"} image of the job by {"name", "image": base64}. Returns how many."""
    images = (job.get("input") or {}).get("images")
    if not isinstance(images, list):
        return 0
    count = 0
    for entry in images:
        if not isinstance(entry, dict) or entry.get("image") or not entry.get("url"):
            continue
        url = str(entry["url"])
        name = str(entry.get("name") or "")
        if urlparse(url).scheme != "https":
            raise RemoteImageError(f"Image {name or '?'}: only https addresses are accepted")
        if status:
            status(name)
        try:
            data, content_type = fetch(url)
        except RemoteImageError:
            raise
        except Exception as error:  # the address expired, the gateway refused it, the network failed
            raise RemoteImageError(f"Image {name or '?'} could not be downloaded: {error}") from error
        if len(data) > MAX_BYTES:
            raise RemoteImageError(f"Image {name or '?'} is larger than {MAX_BYTES // (1024 * 1024)} MB")
        kind = _sniff(data) or (content_type if content_type in IMAGE_TYPES else None)
        if kind is None:
            raise RemoteImageError(f"Image {name or '?'} is not a PNG, JPEG or WebP file")
        entry["image"] = f"data:{kind};base64," + base64.b64encode(data).decode("ascii")
        entry.pop("url", None)
        count += 1
    return count
