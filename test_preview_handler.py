import importlib
import json
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class PreviewHandlerIntegrationTests(unittest.TestCase):
    def test_wrapper_preserves_upstream_result_and_publishes_preview(self):
        jpeg = b"\xff\xd8\xffpreview"
        messages = [
            json.dumps(
                {"type": "progress", "data": {"value": 5, "max": 20}}
            ),
            struct.pack(">II", 1, 1) + jpeg,
        ]

        class FakeWebSocket:
            def __init__(self):
                self.messages = list(messages)

            def recv(self, *args, **kwargs):
                return self.messages.pop(0)

        fake_websocket = types.ModuleType("websocket")
        fake_websocket.WebSocket = FakeWebSocket

        updates = []
        fake_runpod = types.ModuleType("runpod")
        fake_runpod.serverless = types.SimpleNamespace(
            progress_update=lambda job, payload: updates.append((job, payload)),
            start=lambda config: None,
        )

        upstream_source = """
import websocket

def get_history(prompt_id):
    return {
        prompt_id: {
            "outputs": {
                "807": {
                    "text": ["Enhanced prompt from Qwen"],
                    "negative_prompt": ["cgi, plastic skin"],
                    "style_preset": ["editorial"],
                }
            }
        }
    }

def handler(job):
    socket = websocket.WebSocket()
    socket.recv()
    socket.recv()
    get_history("prompt-id")
    return {"images": [{"filename": "final.png"}]}
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            upstream_path = Path(temp_dir) / "handler.py"
            upstream_path.write_text(upstream_source, encoding="utf-8")

            with patch.dict(
                sys.modules,
                {"runpod": fake_runpod, "websocket": fake_websocket},
            ), patch.dict(
                os.environ,
                {
                    "POLLEN_UPSTREAM_HANDLER_PATH": str(upstream_path),
                    "POLLEN_PREVIEW_ENABLED": "true",
                    "POLLEN_PREVIEW_INTERVAL_MS": "1",
                },
                clear=False,
            ):
                sys.modules.pop("preview_handler", None)
                module = importlib.import_module("preview_handler")
                job = {
                    "id": "test",
                    "input": {
                        "workflow": {
                            "807": {
                                "class_type": "LLMPromptEnhancer",
                                "inputs": {"style_preset": "editorial"},
                            }
                        }
                    },
                }
                result = module.handler(job)
                sys.modules.pop("preview_handler", None)

        self.assertEqual(result["images"][0]["filename"], "final.png")
        self.assertEqual(result["enhancedPrompt"], "Enhanced prompt from Qwen")
        self.assertEqual(result["enhancedNegativePrompt"], "cgi, plastic skin")
        self.assertEqual(result["enhancerPreset"], "editorial")
        self.assertEqual(len(updates), 3)
        self.assertEqual(updates[0][0], job)
        self.assertEqual(updates[0][1]["stage"], "preparing_worker")
        self.assertEqual(updates[1][1]["step"], 5)
        self.assertEqual(updates[1][1]["total"], 20)
        self.assertTrue(
            updates[1][1]["previewImage"].startswith(
                "data:image/jpeg;base64,"
            )
        )
        self.assertEqual(updates[2][1]["stage"], "finalizing")


if __name__ == "__main__":
    unittest.main()
