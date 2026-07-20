import base64
import json
import os
import struct
import unittest
from unittest.mock import patch

from preview_bridge import (
    PreviewBridge,
    decode_binary_preview,
    extract_prompt_enhancer_metadata,
    extract_progress,
    stage_for_node,
)


JPEG = b"\xff\xd8\xffpreview-jpeg"
PNG = b"\x89PNG\r\n\x1a\npreview-png"


class PreviewProtocolTests(unittest.TestCase):
    def test_decodes_legacy_jpeg(self):
        message = struct.pack(">II", 1, 1) + JPEG
        self.assertEqual(decode_binary_preview(message), ("image/jpeg", JPEG))

    def test_decodes_legacy_png(self):
        message = struct.pack(">II", 1, 2) + PNG
        self.assertEqual(decode_binary_preview(message), ("image/png", PNG))

    def test_decodes_metadata_preview(self):
        metadata = json.dumps({"image_type": "image/png"}).encode()
        message = struct.pack(">II", 4, len(metadata)) + metadata + PNG
        self.assertEqual(decode_binary_preview(message), ("image/png", PNG))

    def test_extracts_sampler_progress(self):
        message = json.dumps(
            {"type": "progress", "data": {"value": 12, "max": 45}}
        )
        self.assertEqual(extract_progress(message), (12, 45))


class PreviewBridgeTests(unittest.TestCase):
    def test_extracts_prompt_enhancer_outputs_from_history(self):
        job = {
            "input": {
                "workflow": {
                    "807": {
                        "class_type": "LLMPromptEnhancer",
                        "inputs": {"style_preset": "ambrojo_bw"},
                    }
                }
            }
        }
        history = {
            "prompt-id": {
                "outputs": {
                    "807": {
                        "text": ["Enhanced editorial description"],
                        "negative_prompt": ["cgi, plastic skin"],
                        "style_preset": ["ambrojo_bw"],
                    }
                }
            }
        }
        self.assertEqual(
            extract_prompt_enhancer_metadata(job, history),
            {
                "enhancedPrompt": "Enhanced editorial description",
                "enhancedNegativePrompt": "cgi, plastic skin",
                "enhancerPreset": "ambrojo_bw",
            },
        )

    def test_publishes_data_uri_and_throttles(self):
        sent = []
        now = [10.0]
        job = {
            "id": "test-job",
            "input": {"workflow": {"1": {"inputs": {"steps": 45}}}},
        }

        with patch.dict(
            os.environ,
            {"POLLEN_PREVIEW_INTERVAL_MS": "750"},
            clear=False,
        ):
            bridge = PreviewBridge(
                job,
                lambda current_job, payload: sent.append((current_job, payload)),
                clock=lambda: now[0],
            )
            bridge.observe(
                json.dumps(
                    {"type": "progress", "data": {"value": 8, "max": 45}}
                )
            )
            frame = struct.pack(">II", 1, 1) + JPEG
            bridge.observe(frame)
            bridge.observe(frame)
            now[0] += 0.8
            bridge.observe(frame)

        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[0][0], job)
        self.assertEqual(sent[0][1]["step"], 8)
        self.assertEqual(sent[0][1]["total"], 45)
        self.assertEqual(sent[0][1]["stage"], "sampling")
        expected = base64.b64encode(JPEG).decode("ascii")
        self.assertEqual(
            sent[1][1]["previewImage"], f"data:image/jpeg;base64,{expected}"
        )

    def test_reports_loader_and_face_detailer_nodes(self):
        job = {
            "input": {
                "workflow": {
                    "808": {
                        "class_type": "CLIPLoader",
                        "inputs": {},
                        "_meta": {"title": "Load CLIP"},
                    },
                    "900": {
                        "class_type": "PollenFaceDetailerAutoRetry",
                        "inputs": {},
                        "_meta": {"title": "Face Detail"},
                    },
                }
            }
        }
        self.assertEqual(
            stage_for_node(job, "808"),
            ("loading_models", "Loading models", "Load CLIP"),
        )
        self.assertEqual(
            stage_for_node(job, "900"),
            ("refining_face", "Refining face", "Face Detail"),
        )

    def test_reports_prompt_enhancer_before_model_loading(self):
        job = {
            "input": {
                "workflow": {
                    "807": {
                        "class_type": "LLMPromptEnhancer",
                        "inputs": {},
                        "_meta": {"title": "LLM Prompt Enhancer (GPU)"},
                    },
                }
            }
        }
        self.assertEqual(
            stage_for_node(job, "807"),
            (
                "enhancing_prompt",
                "Enhancing prompt · Qwen 8B",
                "LLM Prompt Enhancer (GPU)",
            ),
        )

    def test_reports_shark_sampler_as_sampling(self):
        job = {
            "input": {
                "workflow": {
                    "952": {
                        "class_type": "SharkSampler_Beta",
                        "inputs": {"steps": 40},
                        "_meta": {"title": "SharkSampler"},
                    }
                }
            }
        }
        self.assertEqual(
            stage_for_node(job, "952"),
            ("sampling", "Generating image", "SharkSampler"),
        )

    def test_executing_event_publishes_node_stage(self):
        sent = []
        job = {
            "input": {
                "workflow": {
                    "1": {
                        "class_type": "VAEDecode",
                        "inputs": {},
                        "_meta": {"title": "Decode"},
                    }
                }
            }
        }
        bridge = PreviewBridge(job, lambda current_job, payload: sent.append(payload))
        bridge.observe(json.dumps({"type": "executing", "data": {"node": "1"}}))

        self.assertEqual(sent[-1]["stage"], "decoding")
        self.assertEqual(sent[-1]["stageLabel"], "Decoding image")
        self.assertEqual(sent[-1]["detail"], "Decode")


if __name__ == "__main__":
    unittest.main()
