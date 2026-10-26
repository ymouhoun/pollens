import json
import os
import tempfile
import unittest

import pollen_timings
from pollen_timings import JobTimer, container_started_at, node_category


class FakeClock:
    def __init__(self, start=1_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


WORKFLOW = {
    "1": {"class_type": "UNETLoader", "_meta": {"title": "Load Diffusion Model"}},
    "2": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive"}},
    "3": {"class_type": "ClownsharKSampler_Beta", "_meta": {"title": "Sampler"}},
    "4": {"class_type": "VAEDecode", "_meta": {"title": "Decode"}},
    "5": {"class_type": "SaveImage", "_meta": {"title": "Save"}},
}


def msg(kind, **data):
    return json.dumps({"type": kind, "data": data})


class NodeCategoryTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(node_category("UNETLoader"), "load")
        self.assertEqual(node_category("LoraLoaderModelOnly"), "load")
        self.assertEqual(node_category("UltralyticsDetectorProvider"), "load")
        self.assertEqual(node_category("CLIPTextEncode"), "encode")
        self.assertEqual(node_category("TextEncodeQwenImageEdit"), "encode")
        self.assertEqual(node_category("LLMPromptEnhancer"), "enhance")
        self.assertEqual(node_category("ClownsharKSampler_Beta"), "sample")
        self.assertEqual(node_category("PollenFaceDetailerAutoRetry"), "face")
        self.assertEqual(node_category("ImageUpscaleWithModel"), "upscale")
        self.assertEqual(node_category("UpscaleModelLoader"), "load")
        self.assertEqual(node_category("VAEDecode"), "decode")
        self.assertEqual(node_category("SaveImage"), "save")
        self.assertEqual(node_category("FluxResolutionNode"), "other")


class ContainerStartTests(unittest.TestCase):
    def test_reads_pid1_start_from_proc(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "1"))
            with open(os.path.join(root, "stat"), "w") as f:
                f.write("cpu  1 2 3\nbtime 1700000000\nprocesses 5\n")
            ticks = os.sysconf("SC_CLK_TCK")
            fields = ["S"] + ["0"] * 18 + [str(ticks * 42)] + ["0"] * 10
            with open(os.path.join(root, "1", "stat"), "w") as f:
                f.write("1 (python (boot)) " + " ".join(fields))
            self.assertAlmostEqual(container_started_at(root), 1700000042.0)

    def test_missing_proc_is_none(self):
        self.assertIsNone(container_started_at("/nonexistent-proc"))


class JobTimerTests(unittest.TestCase):
    def setUp(self):
        pollen_timings._GPU_NAME[:] = ["NVIDIA RTX PRO 6000 Blackwell"]

    def run_job(self, cached=()):
        clock = FakeClock()
        timer = JobTimer(
            {"input": {"workflow": WORKFLOW}},
            clock=clock,
            boot={"bootstrapStartedAt": 990.0, "linksReadyAt": 990.5, "revision": "abc", "checkpoints": ["editorial10_base.safetensors"]},
            container_start=985.0,
        )
        timer.timed("comfyWait", lambda: clock.advance(4))
        timer.timed("queue", lambda: clock.advance(0.1))
        timer.observe(msg("execution_start", prompt_id="p"))
        if cached:
            timer.observe(msg("execution_cached", nodes=list(cached)))
        if "1" not in cached:
            timer.observe(msg("executing", node="1"))
            clock.advance(12)
        timer.observe(msg("executing", node="2"))
        clock.advance(3)
        timer.observe(msg("executing", node="3"))
        clock.advance(8)  # model into VRAM before the first step
        timer.observe(msg("progress", value=1, max=40))
        clock.advance(30)
        timer.observe(msg("progress", value=40, max=40))
        timer.observe(msg("executing", node="4"))
        clock.advance(1.5)
        timer.observe(msg("executing", node="5"))
        clock.advance(0.5)
        timer.observe(msg("executing", node=None))
        clock.advance(2)  # history, image download, base64
        return timer.finish()

    def test_cold_job_split(self):
        out = self.run_job()
        p = out["phases"]
        self.assertEqual(p["comfyWait"], 4000)
        self.assertEqual(p["queue"], 100)
        self.assertEqual(p["load"], 12000)
        self.assertEqual(p["encode"], 3000)
        self.assertEqual(p["vram"], 8000)
        self.assertEqual(p["sample"], 30000)
        self.assertEqual(p["decode"], 1500)
        self.assertEqual(p["save"], 500)
        self.assertEqual(p["output"], 2000)
        self.assertEqual(out["steps"], 40)
        self.assertEqual(out["worker"]["containerStartedAt"], 985000)
        self.assertEqual(out["worker"]["linksReadyAt"], 990500)
        self.assertEqual(out["worker"]["checkpoints"], ["editorial10_base.safetensors"])
        self.assertEqual(out["worker"]["gpu"], "NVIDIA RTX PRO 6000 Blackwell")
        self.assertEqual(out["job"]["receivedAt"], 1000000)
        sampler = next(n for n in out["nodes"] if n["id"] == "3")
        self.assertEqual(sampler["toFirstStepMs"], 8000)
        self.assertEqual(sampler["ms"], 38000)

    def test_cached_loader_costs_nothing(self):
        out = self.run_job(cached=("1",))
        self.assertNotIn("load", out["phases"])
        self.assertEqual(out["cached"], ["1"])

    def test_job_index_counts_jobs_of_the_process(self):
        first = self.run_job()["worker"]["jobIndex"]
        second = self.run_job()["worker"]["jobIndex"]
        self.assertEqual(second, first + 1)

    def test_ignores_binary_and_garbage(self):
        timer = JobTimer({"input": {}}, clock=FakeClock(), boot={}, container_start=None)
        timer.observe(b"\x00\x00\x00\x01")
        timer.observe("not json")
        timer.observe(json.dumps([1, 2]))
        out = timer.finish()
        self.assertEqual(out["phases"], {})

    def test_boot_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "boot.json")
            pollen_timings.write_boot_file({"revision": "r1"}, path)
            self.assertEqual(pollen_timings.read_boot_file(path), {"revision": "r1"})
            self.assertEqual(pollen_timings.read_boot_file(os.path.join(d, "missing.json")), {})


if __name__ == "__main__":
    unittest.main()
