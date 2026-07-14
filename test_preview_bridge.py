import base64
import json
import os
import struct
import unittest
from unittest.mock import patch

from preview_bridge import PreviewBridge, decode_binary_preview, extract_progress


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

        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0][0], job)
        self.assertEqual(sent[0][1]["step"], 8)
        self.assertEqual(sent[0][1]["total"], 45)
        expected = base64.b64encode(JPEG).decode("ascii")
        self.assertEqual(
            sent[0][1]["previewImage"], f"data:image/jpeg;base64,{expected}"
        )


if __name__ == "__main__":
    unittest.main()
