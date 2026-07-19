"""FaceDetailer wrapper that retries only near-black refined patches."""

from __future__ import annotations

import os
import threading

from impact.hooks import (
    BlackPatchRetryHook,
    DetailerHookCombine,
)
from impact.impact_pack import (
    DetailerForEach,
    DetailerForEachAutoRetry,
    FaceDetailer,
)


_PATCH_LOCK = threading.RLock()


def _env_int(name: str, fallback: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, fallback))
    except (TypeError, ValueError):
        value = fallback
    return max(minimum, min(maximum, value))


class PollenFaceDetailerAutoRetry:
    """Run the original Impact Pack FaceDetailer with one black-patch retry."""

    RETURN_TYPES = FaceDetailer.RETURN_TYPES
    RETURN_NAMES = FaceDetailer.RETURN_NAMES
    OUTPUT_IS_LIST = FaceDetailer.OUTPUT_IS_LIST
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Simple"
    DESCRIPTION = (
        "The original Impact Pack FaceDetailer with identical inputs and "
        "a targeted retry when the refined patch is nearly black."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return FaceDetailer.INPUT_TYPES()

    def doit(self, **kwargs):
        mean_threshold = _env_int(
            "POLLEN_BLACK_PATCH_MEAN_THRESHOLD", 10, 0, 255
        )
        variance_threshold = _env_int(
            "POLLEN_BLACK_PATCH_VARIANCE_THRESHOLD", 5, 0, 255
        )
        max_attempts = _env_int("POLLEN_FACE_DETAIL_MAX_ATTEMPTS", 2, 1, 3)

        retry_hook = BlackPatchRetryHook(mean_threshold, variance_threshold)
        existing_hook = kwargs.get("detailer_hook")
        kwargs["detailer_hook"] = (
            DetailerHookCombine(existing_hook, retry_hook)
            if existing_hook is not None
            else retry_hook
        )

        # FaceDetailer delegates its sampling to DetailerForEach. Replace only
        # that call while this node executes, then restore the original method.
        original_descriptor = DetailerForEach.__dict__["do_detail"]

        def retrying_do_detail(*args, **inner_kwargs):
            inner_kwargs["max_retries"] = max_attempts
            return DetailerForEachAutoRetry.do_detail(*args, **inner_kwargs)

        with _PATCH_LOCK:
            DetailerForEach.do_detail = staticmethod(retrying_do_detail)
            try:
                return FaceDetailer().doit(**kwargs)
            finally:
                DetailerForEach.do_detail = original_descriptor


NODE_CLASS_MAPPINGS = {
    "PollenFaceDetailerAutoRetry": PollenFaceDetailerAutoRetry,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PollenFaceDetailerAutoRetry": "FaceDetailer (Pollen auto retry)",
}

