import tempfile
import unittest
from pathlib import Path

from dp_manip.failure_round2 import choose_candidate, paired_changes, write_once


def result(values):
    return {"episodes": [{"seed": s, "success_once": ok} for s, ok in values]}


class Round2Tests(unittest.TestCase):
    def test_video_export_into_absent_nested_directory(self):
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401
            import numpy as np
            from PIL import Image
        except ModuleNotFoundError:
            self.skipTest("media dependencies are installed on the cluster")
        from scripts.failure_round2 import write_episode_video

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "round2/diagnostics/replay.mp4"
            frames = np.zeros((12, 32, 32, 3), dtype=np.uint8)
            write_episode_video(target, frames)
            self.assertGreater(target.stat().st_size, 0)
            with imageio.get_reader(target) as video:
                self.assertEqual(video.get_data(0).shape, (32, 32, 3))
            with Image.open(target.with_suffix(".png")) as sheet:
                self.assertEqual(sheet.size, (128, 156))

    def test_pairing_uses_seed_not_position(self):
        change = paired_changes(result([(1, True), (2, False), (3, True)]),
                                result([(3, True), (2, True), (1, False)]))
        self.assertEqual(change["gained"], [1])
        self.assertEqual(change["lost"], [2])
        self.assertEqual(change["difference"], 0)

    def test_invalid_pairing_refused(self):
        for bad in ([(1, True), (1, False)], [(2, True)]):
            with self.assertRaises(ValueError):
                paired_changes(result(bad), result([(1, True)]))

    def test_tie_rule_is_not_maximum_only(self):
        rows = [{"successes": 19, "alpha": 1.0, "steps": 5000},
                {"successes": 17, "alpha": 0.25, "steps": 10000},
                {"successes": 17, "alpha": 0.25, "steps": 5000},
                {"successes": 16, "alpha": 0.1, "steps": 5000}]
        self.assertEqual(choose_candidate(rows), rows[2])

    def test_decisions_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            write_once(path, {"alpha": 0.5})
            write_once(path, {"alpha": 0.5})
            with self.assertRaises(ValueError):
                write_once(path, {"alpha": 1.0})


if __name__ == "__main__":
    unittest.main()
