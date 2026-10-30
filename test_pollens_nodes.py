"""The colour-matching node of the product edit (pollens_nodes). Needs torch: skipped where it is missing."""
import unittest

try:
    import torch
except ImportError:  # the light test environment
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class MatchGradeTests(unittest.TestCase):
    def setUp(self):
        from pollens_nodes import PollensMatchGrade, _to_lab, _to_rgb
        self.node, self.to_lab, self.to_rgb = PollensMatchGrade(), _to_lab, _to_rgb

    def test_lab_round_trip(self):
        x = torch.rand(1, 32, 32, 3)
        self.assertLess((self.to_rgb(self.to_lab(x)) - x).abs().max().item(), 1e-4)

    def test_brings_a_saturated_item_back_and_leaves_the_rest(self):
        torch.manual_seed(0)
        h, w = 96, 64
        original = torch.full((1, h, w, 3), 0.55)
        original[0, 20:80, 16:48] = torch.tensor([0.62, 0.18, 0.2]) + 0.03 * torch.randn(60, 32, 3)
        mask = torch.zeros(1, h, w)
        mask[0, 20:80, 16:48] = 1
        edited = original.clone()
        edited[0, 20:80, 16:48] = torch.tensor([0.85, 0.05, 0.12]) + 0.03 * torch.randn(60, 32, 3)
        out, report = self.node.run(edited.clamp(0, 1), original.clamp(0, 1), mask, mask, 1.0, 1.0, 0.0, 2, 0)
        lab = lambda t: self.to_lab(t[0, 30:70, 20:44])
        chroma = lambda l: (l[..., 1] ** 2 + l[..., 2] ** 2).sqrt().mean().item()
        self.assertLess(abs(chroma(lab(out)) - chroma(lab(original.clamp(0, 1)))), abs(chroma(lab(edited.clamp(0, 1))) - chroma(lab(original.clamp(0, 1)))) / 4)
        self.assertTrue(torch.equal(out[0, :10], edited.clamp(0, 1)[0, :10]))
        self.assertIn("saturation", report)

    def test_tiny_mask_changes_nothing(self):
        img = torch.rand(1, 32, 32, 3)
        mask = torch.zeros(1, 32, 32)
        out, report = self.node.run(img, img, mask, mask, 0.5, 0.8, 0.0, 6, 8)
        self.assertTrue(torch.equal(out, img))
        self.assertIn("inchangée", report)


if __name__ == "__main__":
    unittest.main()
