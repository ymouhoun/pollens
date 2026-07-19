import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from face_asset_cache import (
    FaceAssetCache,
    ensure_face_dependencies,
    safe_relative_path,
)


class FaceAssetCacheTests(unittest.TestCase):
    def make_cache(self, root: Path, *, max_loras: int = 2):
        calls = []

        def downloader(**kwargs):
            calls.append(kwargs["filename"])
            destination = Path(kwargs["cache_dir"]) / "downloaded"
            destination.mkdir(parents=True, exist_ok=True)
            file_path = destination / Path(kwargs["filename"]).name
            file_path.write_bytes(kwargs["filename"].encode())
            return str(file_path)

        cache = FaceAssetCache(
            repo_id="ymouhoun/faces",
            token="test-token",
            revision="main",
            cache_root=root / "cache",
            comfy_models_root=root / "models",
            max_loras=max_loras,
            downloader=downloader,
        )
        return cache, calls

    def test_reuses_cached_file_without_downloading_again(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache, calls = self.make_cache(Path(temp_dir))
            statuses = []
            first = cache.ensure(
                source="loras/carlotta.safetensors",
                target="loras/carlotta.safetensors",
                pinned=False,
                status_callback=lambda state, source: statuses.append((state, source)),
            )
            second = cache.ensure(
                source="loras/carlotta.safetensors",
                target="loras/carlotta.safetensors",
                pinned=False,
                status_callback=lambda state, source: statuses.append((state, source)),
            )

            self.assertEqual(first, second)
            self.assertTrue(first.is_symlink())
            self.assertEqual(calls, ["loras/carlotta.safetensors"])
            self.assertEqual(
                statuses,
                [
                    ("downloading", "loras/carlotta.safetensors"),
                    ("cached", "loras/carlotta.safetensors"),
                ],
            )

    def test_lru_prunes_old_lora_but_keeps_pinned_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache, _ = self.make_cache(Path(temp_dir), max_loras=2)
            cache.ensure(
                source="assets/sams/sam.pth",
                target="sams/sam.pth",
                pinned=True,
            )
            cache.ensure(
                source="loras/one.safetensors",
                target="loras/one.safetensors",
                pinned=False,
            )
            time.sleep(0.01)
            cache.ensure(
                source="loras/two.safetensors",
                target="loras/two.safetensors",
                pinned=False,
            )
            time.sleep(0.01)
            cache.ensure(
                source="loras/three.safetensors",
                target="loras/three.safetensors",
                pinned=False,
            )

            models = Path(temp_dir) / "models"
            self.assertFalse((models / "loras/one.safetensors").exists())
            self.assertTrue((models / "loras/two.safetensors").is_symlink())
            self.assertTrue((models / "loras/three.safetensors").is_symlink())
            self.assertTrue((models / "sams/sam.pth").is_symlink())

    def test_rejects_path_traversal(self):
        with self.assertRaises(RuntimeError):
            safe_relative_path("../secret.safetensors")

    def test_face_job_uses_environment_configuration(self):
        ensured = []

        class FakeCache:
            @classmethod
            def from_environment(cls):
                return cls()

            def ensure(self, **kwargs):
                ensured.append(kwargs)

        job = {
            "input": {
                "workflow": {},
                "faceLora": "loras/carlotta.safetensors",
            }
        }
        with patch("face_asset_cache.FaceAssetCache", FakeCache), patch.dict(
            os.environ,
            {
                "POLLEN_FACE_ASSETS_JSON": (
                    '[{"source":"assets/sam.pth","target":"sams/sam.pth"}]'
                )
            },
            clear=False,
        ):
            ensure_face_dependencies(job)

        self.assertEqual(ensured[0]["source"], "assets/sam.pth")
        self.assertTrue(ensured[0]["pinned"])
        self.assertEqual(ensured[1]["target"], "loras/carlotta.safetensors")
        self.assertFalse(ensured[1]["pinned"])


if __name__ == "__main__":
    unittest.main()
