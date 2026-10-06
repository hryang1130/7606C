from __future__ import annotations

import json
import unittest
from pathlib import Path

from dp_manip.config import from_recorded, load


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baselines" / "phase0" / "pickcube_rgb.json"


class PhaseZeroBaselineTest(unittest.TestCase):
    def test_canonical_config_matches_frozen_reference(self) -> None:
        reference = json.loads(MANIFEST.read_text(encoding="utf-8"))
        resolved = load(ROOT / "configs" / "tasks" / "pickcube.toml")
        self.assertEqual(resolved, from_recorded(reference["canonical_config"]))

    def test_regression_override_matches_captured_budget(self) -> None:
        reference = json.loads(MANIFEST.read_text(encoding="utf-8"))
        resolved = load(
            ROOT / "configs" / "tasks" / "pickcube.toml",
            experiment=ROOT / "configs" / "regression" / "pickcube_phase0.toml",
        )
        for section, expected in reference["regression_overrides"].items():
            actual = getattr(resolved, section)
            for key, value in expected.items():
                self.assertEqual(getattr(actual, key), value)

    def test_captured_artifact_fingerprints_are_complete(self) -> None:
        reference = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(len(reference["reference_commit"]), 40)
        for split in ("train", "validation"):
            self.assertEqual(len(reference["dataset"][split]["sha256"]), 64)
        checkpoints = reference["observed"]["checkpoint_sha256"]
        self.assertEqual(
            set(checkpoints),
            {"final.pt", "resume.pt", "step_000001.pt", "step_000005.pt"},
        )
        self.assertTrue(all(len(value) == 64 for value in checkpoints.values()))


if __name__ == "__main__":
    unittest.main()
