import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap
import pollen_timings


class BootstrapTimingsTests(unittest.TestCase):
    def test_links_and_boot_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            snapshot = root / "hub" / "models--ymouhoun--pollens-edito" / "snapshots" / "rev123"
            (snapshot / "weights").mkdir(parents=True)
            (snapshot / "weights" / "editorial10_base.safetensors").write_bytes(b"x")
            (snapshot / "weights" / "vae.safetensors").write_bytes(b"x")
            (snapshot / "comfy-models.json").write_text(json.dumps({"files": [
                {"source": "weights/editorial10_base.safetensors", "target": "diffusion_models/editorial10_base.safetensors"},
                {"source": "weights/vae.safetensors", "target": "vae/qwen_image_vae.safetensors"},
            ]}))
            boot_file = root / "boot.json"
            with patch.object(bootstrap, "COMFY_MODELS_ROOT", root / "models"), \
                 patch.object(pollen_timings, "BOOT_FILE", str(boot_file)):
                targets = bootstrap.create_model_links(snapshot)
                bootstrap.write_boot_timings("ymouhoun/pollens-edito", snapshot, targets)
            self.assertTrue((root / "models" / "diffusion_models" / "editorial10_base.safetensors").is_symlink())
            boot = json.loads(boot_file.read_text())
            self.assertEqual(boot["revision"], "rev123")
            self.assertEqual(boot["modelId"], "ymouhoun/pollens-edito")
            self.assertEqual(boot["checkpoints"], ["editorial10_base.safetensors"])
            self.assertLessEqual(boot["bootstrapStartedAt"], boot["linksReadyAt"])


if __name__ == "__main__":
    unittest.main()
