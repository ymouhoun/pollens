import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("pollen_face_detailer_retry.py")


class PollenFaceDetailerRetryTests(unittest.TestCase):
    def _load_module(self, retry_error=None):
        calls = []
        fallback_calls = []

        class FakeRetryHook:
            def __init__(self, mean_threshold, variance_threshold):
                self.mean_threshold = mean_threshold
                self.variance_threshold = variance_threshold

        class FakeHookCombine:
            def __init__(self, first, second):
                self.first = first
                self.second = second

        class FakeDetailerForEach:
            @staticmethod
            def do_detail(*args, **kwargs):
                fallback_calls.append(kwargs)
                return (args[0],)

        class FakeDetailerForEachAutoRetry:
            @staticmethod
            def do_detail(*args, **kwargs):
                calls.append(kwargs)
                if retry_error is not None:
                    raise retry_error
                return ("retried",)

        class FakeFaceDetailer:
            RETURN_TYPES = ("IMAGE",)
            RETURN_NAMES = ("image",)
            OUTPUT_IS_LIST = (False,)

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)}}

            def doit(self, **kwargs):
                return FakeDetailerForEach.do_detail(
                    kwargs["image"],
                    model=kwargs.get("model"),
                    detailer_hook=kwargs.get("detailer_hook"),
                )

        impact = types.ModuleType("impact")
        hooks = types.ModuleType("impact.hooks")
        impact_pack = types.ModuleType("impact.impact_pack")
        hooks.BlackPatchRetryHook = FakeRetryHook
        hooks.DetailerHookCombine = FakeHookCombine
        impact_pack.DetailerForEach = FakeDetailerForEach
        impact_pack.DetailerForEachAutoRetry = FakeDetailerForEachAutoRetry
        impact_pack.FaceDetailer = FakeFaceDetailer

        modules = {
            "impact": impact,
            "impact.hooks": hooks,
            "impact.impact_pack": impact_pack,
        }
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                "pollen_face_detailer_retry_under_test", MODULE_PATH
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        return module, calls, fallback_calls, FakeDetailerForEach, FakeRetryHook

    def test_retries_with_same_face_detailer_inputs_and_restores_original(self):
        module, calls, _, detailer, retry_hook = self._load_module()
        original_descriptor = detailer.__dict__["do_detail"]

        with patch.dict(os.environ, {}, clear=False):
            result = module.PollenFaceDetailerAutoRetry().doit(image="source")

        self.assertEqual(result, ("retried",))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_retries"], 2)
        self.assertIsInstance(calls[0]["detailer_hook"], retry_hook)
        self.assertIs(detailer.__dict__["do_detail"], original_descriptor)

    def test_exposes_the_original_face_detailer_schema(self):
        module, _, _, _, _ = self._load_module()
        self.assertEqual(
            module.PollenFaceDetailerAutoRetry.INPUT_TYPES(),
            {"required": {"image": ("IMAGE",)}},
        )

    def test_returns_source_image_when_all_patches_are_invalid(self):
        module, calls, fallback_calls, detailer, _ = self._load_module(
            RuntimeError("Max retries reached")
        )
        original_descriptor = detailer.__dict__["do_detail"]

        with patch("builtins.print") as print_mock:
            result = module.PollenFaceDetailerAutoRetry().doit(
                image="source", model="loaded-model"
            )

        self.assertEqual(result, ("source",))
        self.assertEqual(len(calls), 1)
        self.assertEqual(fallback_calls[-1]["model"], "DUMMY")
        self.assertIsNone(fallback_calls[-1]["detailer_hook"])
        self.assertIn("2 invalid patches", print_mock.call_args.args[0])
        self.assertIs(detailer.__dict__["do_detail"], original_descriptor)

    def test_does_not_hide_unrelated_face_detail_errors(self):
        module, _, _, detailer, _ = self._load_module(
            RuntimeError("CUDA allocation failed")
        )
        original_descriptor = detailer.__dict__["do_detail"]

        with self.assertRaisesRegex(RuntimeError, "CUDA allocation failed"):
            module.PollenFaceDetailerAutoRetry().doit(image="source")

        self.assertIs(detailer.__dict__["do_detail"], original_descriptor)


if __name__ == "__main__":
    unittest.main()
