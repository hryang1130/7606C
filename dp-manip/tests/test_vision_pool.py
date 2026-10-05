"""Switchable vision pooling: average pooling vs robomimic spatial softmax.

``vision.pool`` selects how the ResNet-18 layer4 map becomes the per-camera
feature. Both heads return ``feature_dim`` features, older recorded configs
resolve to average pooling, and the ``vision_pool`` matrix differs only in the
declared variable.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from dp_manip import config as config_lib
from dp_manip.config import default_run_name, from_recorded, load, load_experiment
from dp_manip.invariants import allowed_keys, config_differences, control_hash
from test_backbone_interface import make_stats
from test_checkpoint_lifecycle import smoke_config

try:
    import torch
    import torch.nn as nn

    from dp_manip.policy import DiffusionPolicy
    from dp_manip.trainer import run_training
    from dp_manip.vision import ResNet18Encoder, SpatialSoftmax
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "configs" / "tasks"
EXPERIMENT = ROOT / "configs" / "experiments" / "vision_pool.toml"

# Parameter names of the average-pooling encoder before vision.pool existed.
AVG_HEAD_KEYS = {"head.2.weight", "head.2.bias"}


class VisionPoolConfigTest(unittest.TestCase):
    def test_baseline_keeps_average_pooling(self) -> None:
        vision = load(TASKS / "pickcube.toml").vision
        self.assertEqual(vision.pool, "avg")
        self.assertEqual(vision.num_keypoints, 32)

    def test_invalid_pool_settings_are_rejected(self) -> None:
        for override, message in (
            ('vision.pool="max"', "vision.pool"),
            ('vision.pool="AVG"', "vision.pool"),
            ("vision.pool=1", "vision.pool"),
            ("vision.num_keypoints=0", "vision.num_keypoints"),
            ("vision.num_keypoints=true", "vision.num_keypoints"),
            ("vision.num_keypoints=1.5", "vision.num_keypoints"),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                load(TASKS / "pickcube.toml", [override])

    def test_recorded_configs_without_pool_resolve_to_average_pooling(self) -> None:
        current = load(TASKS / "pickcube.toml").to_dict()
        recorded = load(TASKS / "pickcube.toml").to_dict()
        del recorded["vision"]["pool"]
        del recorded["vision"]["num_keypoints"]
        restored = from_recorded(recorded)
        self.assertEqual(restored.vision.pool, "avg")
        self.assertEqual(restored.vision.num_keypoints, 32)
        self.assertEqual(restored.to_dict(), current)
        with self.assertRaisesRegex(ValueError, "pool"):
            config_lib.from_dict(recorded)

    def test_run_names(self) -> None:
        task = TASKS / "peginsertionside.toml"
        avg = load(task, ["data.num_demos=200", "train.seed=1"])
        # Average pooling keeps every existing run directory name.
        self.assertEqual(default_run_name(avg), "peginsertionside_rgb_unet_n200_s1")
        spatial = load(task, ["data.num_demos=200", "train.seed=1", 'vision.pool="spatial_softmax"'])
        self.assertEqual(default_run_name(spatial), "peginsertionside_rgb_unet_ss32_n200_s1")
        fewer = load(
            task,
            ["data.num_demos=200", "train.seed=1", 'vision.pool="spatial_softmax"', "vision.num_keypoints=16"],
        )
        self.assertEqual(default_run_name(fewer), "peginsertionside_rgb_unet_ss16_n200_s1")


class VisionPoolExperimentTest(unittest.TestCase):
    def test_spec(self) -> None:
        spec = load_experiment(EXPERIMENT)
        self.assertEqual(spec.variable, "vision.pool")
        self.assertEqual(spec.values, ("avg", "spatial_softmax"))
        self.assertEqual(spec.seeds_for("avg"), (1, 2, 3))
        self.assertEqual(spec.seeds_for("spatial_softmax"), (1, 2, 3))
        self.assertEqual(dict(spec.fixed), {"data.num_demos": 200})
        # num_keypoints is an ordinary control: no exemption beyond the variable.
        self.assertNotIn("vision.num_keypoints", allowed_keys(spec))

    def test_arms_differ_only_in_pool_at_n200(self) -> None:
        spec = load_experiment(EXPERIMENT)
        for task_path in sorted(TASKS.glob("*.toml")):
            with self.subTest(task=task_path.stem):
                avg = load(task_path, ["train.seed=1"], experiment=EXPERIMENT, experiment_value="avg")
                spatial = load(
                    task_path, ["train.seed=1"], experiment=EXPERIMENT, experiment_value="spatial_softmax"
                )
                self.assertEqual(avg.data.num_demos, 200)
                self.assertEqual(
                    config_differences(avg.to_dict(), spatial.to_dict()),
                    {"vision.pool": ("avg", "spatial_softmax")},
                )
                keys = allowed_keys(spec)
                self.assertEqual(control_hash(avg, keys), control_hash(spatial, keys))
                # The avg arm is the data-size N=200 cell, so it reuses that run.
                data_size = load(
                    task_path,
                    ["train.seed=1"],
                    experiment=ROOT / "configs" / "experiments" / "data_size.toml",
                    experiment_value=200,
                )
                self.assertEqual(avg.to_dict(), data_size.to_dict())
                self.assertEqual(default_run_name(avg), default_run_name(data_size))
                self.assertNotEqual(default_run_name(spatial), default_run_name(avg))

    def test_fixed_values_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed to 200"):
            config_lib.load_run(
                TASKS / "peginsertionside.toml",
                experiment=EXPERIMENT,
                experiment_value="spatial_softmax",
                num_demos=100,
            )

    def test_invalid_fixed_tables_are_rejected(self) -> None:
        spec = '[experiment]\nname = "bad"\nvariable = "vision.pool"\nvalues = ["avg"]\n\n[replicates]\n"avg" = [1]\n'
        for text, message in (
            (spec + '[fixed]\n"vision.pool" = "avg"\n', "experiment variable"),
            (spec + '[fixed]\n"num_demos" = 200\n', "section.key"),
            (spec + '[fixed]\n"nope.num_demos" = 200\n', "section.key"),
            ("fixed = 3\n" + spec, r"\[fixed\] must be a table"),
        ):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "experiment.toml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_experiment(path)


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class SpatialSoftmaxEncoderTest(unittest.TestCase):
    def encoder(self, pool: str, num_keypoints: int = 8) -> "ResNet18Encoder":
        torch.manual_seed(0)
        return ResNet18Encoder(16, pool=pool, num_keypoints=num_keypoints)

    def test_both_pools_return_feature_dim(self) -> None:
        images = torch.randn(3, 3, 128, 128)
        for pool in ("avg", "spatial_softmax"):
            with self.subTest(pool=pool):
                self.assertEqual(self.encoder(pool)(images).shape, (3, 16))

    def test_keypoints_are_expected_coordinates_in_unit_square(self) -> None:
        encoder = self.encoder("spatial_softmax")
        self.assertIsInstance(encoder.head[0], SpatialSoftmax)
        feature_map = encoder.feature_map(torch.randn(3, 3, 128, 128))
        # layer4 of a 128x128 image is 4x4 (documented limitation).
        self.assertEqual(feature_map.shape, (3, 512, 4, 4))
        keypoints = encoder.head[0](feature_map * 50.0)
        self.assertEqual(keypoints.shape, (3, 8, 2))
        self.assertTrue(((keypoints >= -1.0) & (keypoints <= 1.0)).all())

    def test_keypoint_head_does_not_start_saturated(self) -> None:
        """Guards the dead-encoder bug of the first vision-pool run.

        ``ResNet18Encoder._initialize`` gives *every* Conv2d
        ``kaiming_normal_(mode="fan_out")``. On the 512 -> K keypoint conv that
        is a weight std of ``sqrt(2 / K)`` (~0.25 here), ten times PyTorch's
        default, which makes the logits large enough for the soft-argmax to
        saturate to a one-hot pick. A saturated soft-argmax has *exactly zero*
        gradient with respect to its logits, so the keypoints freeze and the
        camera branch becomes input-independent: the first pool run measured
        softmax entropy 0.000 of 2.773 nats, keypoint std 0.00000 across 24
        windows from 50 demos, and a constant 128-d feature. See
        ``report/exp_vision_pool_n100.md`` section 5.
        """
        encoder = self.encoder("spatial_softmax", num_keypoints=8)
        head = encoder.head[0]
        images = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            feature_map = encoder.feature_map(images)
            logits = head.keypoints(feature_map)
            attention = torch.softmax(
                logits.float().reshape(4, head.num_keypoints, -1) / head.temperature, dim=-1
            )
            entropy = -(attention * torch.log(attention + 1e-12)).sum(-1).mean()
            keypoints = head(feature_map)
        # Uniform attention over the 4x4 layer4 grid is log(16) = 2.773 nats.
        self.assertLess(float(logits.std()), 2.0)
        self.assertGreater(float(entropy), 1.8)
        # A frozen head maps every image to the same keypoints.
        self.assertGreater(float(keypoints.std(dim=0).mean()), 1e-2)
        self.assertGreaterEqual(float(head.temperature), 0.1)

    def test_keypoint_conv_gets_a_usable_gradient(self) -> None:
        """The keypoint conv must not be gradient-starved relative to its head.

        On the broken run its gradient was ~1/200 of ``head.2``'s, which is the
        signature of a saturated softmax.
        """
        encoder = self.encoder("spatial_softmax", num_keypoints=8)
        encoder(torch.randn(4, 3, 128, 128)).square().mean().backward()
        parameters = dict(encoder.named_parameters())
        keypoint = float(parameters["head.0.keypoints.weight"].grad.norm())
        projection = float(parameters["head.2.weight"].grad.norm())
        self.assertGreater(keypoint, 0.1 * projection)

    def test_keypoint_follows_a_peak(self) -> None:
        pool = SpatialSoftmax(1, 1)
        with torch.no_grad():
            pool.keypoints.weight.fill_(1.0)
            pool.keypoints.bias.zero_()
        feature = torch.zeros(1, 1, 4, 5)
        feature[0, 0, 3, 0] = 100.0  # bottom-left corner
        self.assertTrue(torch.allclose(pool(feature), torch.tensor([[[-1.0, 1.0]]]), atol=1e-4))
        # A uniform map has its expectation at the centre.
        self.assertTrue(torch.allclose(pool(torch.zeros(1, 1, 4, 5)), torch.zeros(1, 1, 2), atol=1e-6))

    def test_softmax_runs_in_fp32_under_autocast(self) -> None:
        encoder = self.encoder("spatial_softmax")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            keypoints = encoder.head[0](encoder.feature_map(torch.randn(2, 3, 64, 64)))
            features = encoder(torch.randn(2, 3, 64, 64))
        self.assertEqual(keypoints.dtype, torch.float32)
        self.assertEqual(features.shape, (2, 16))

    def test_forward_backward_and_optimizer_step(self) -> None:
        encoder = self.encoder("spatial_softmax")
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
        before = {name: value.detach().clone() for name, value in encoder.named_parameters()}
        loss = encoder(torch.randn(4, 3, 64, 64)).square().mean()
        loss.backward()
        for name in ("head.0.keypoints.weight", "head.0.temperature", "head.2.weight"):
            gradient = dict(encoder.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
        optimizer.step()
        after = dict(encoder.named_parameters())
        for name in ("head.0.keypoints.weight", "head.0.temperature", "head.2.weight", "stem.0.weight"):
            self.assertFalse(torch.equal(before[name], after[name]), name)
        self.assertEqual(encoder.head[0].temperature.shape, (1,))

    def test_average_pooling_keeps_its_parameter_names(self) -> None:
        avg = {name for name in self.encoder("avg").state_dict() if name.startswith("head.")}
        self.assertEqual(avg, AVG_HEAD_KEYS)
        spatial = {name for name in self.encoder("spatial_softmax").state_dict() if name.startswith("head.")}
        self.assertEqual(
            spatial,
            {"head.0.keypoints.weight", "head.0.keypoints.bias", "head.0.temperature", "head.2.weight", "head.2.bias"},
        )
        self.assertIsInstance(self.encoder("avg").head[0], nn.AdaptiveAvgPool2d)

    def test_unknown_pool_is_rejected_by_the_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown vision pool"):
            ResNet18Encoder(16, pool="max", num_keypoints=8)


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class VisionPoolCheckpointTest(unittest.TestCase):
    def small_config(self, *overrides: str):
        return load(
            TASKS / "pickcube.toml",
            [
                "vision.feature_dim=8",
                "policy.unet_dims=[16, 32]",
                "policy.kernel_size=3",
                "policy.n_groups=4",
                "diffusion.num_diffusion_iters=4",
                "diffusion.num_inference_iters=2",
                *overrides,
            ],
        )

    def test_checkpoint_without_pool_loads_as_average_pooling(self) -> None:
        cfg = self.small_config()
        torch.manual_seed(0)
        stats = make_stats(5, 4)
        policy = DiffusionPolicy(
            cfg.policy,
            cfg.vision,
            cfg.diffusion,
            image_shape=(64, 64, 3),
            proprio_dim=5,
            action_dim=4,
            stats=stats,
        ).eval()
        recorded = cfg.to_dict()
        del recorded["vision"]["pool"]
        del recorded["vision"]["num_keypoints"]
        checkpoint = {
            "config": recorded,
            "train_data": {"image_shape": [64, 64, 3], "proprio_dim": 5, "action_dim": 4},
            "normalization": stats.to_dict(),
            "model": policy.state_dict(),
        }
        loaded = DiffusionPolicy.from_checkpoint(checkpoint, "cpu").eval()
        head = loaded.observation_encoder.image_encoders[0].head
        self.assertIsInstance(head[0], nn.AdaptiveAvgPool2d)

        rgb = torch.randint(0, 256, (2, policy.obs_horizon, 3, 64, 64), dtype=torch.uint8)
        proprio = torch.randn(2, policy.obs_horizon, 5)
        expected = policy.get_action(rgb, proprio, generator=torch.Generator().manual_seed(0))
        actual = loaded.get_action(rgb, proprio, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(expected, actual))

    def test_spatial_softmax_train_load_sample_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = smoke_config(root / "data")
            cfg = dataclasses.replace(
                base, vision=dataclasses.replace(base.vision, pool="spatial_softmax", num_keypoints=4)
            )
            cfg.validate()
            output_root = root / "runs"
            self.assertEqual(run_training(cfg, output_root=output_root, device="cpu"), 0)
            run_dir = output_root / default_run_name(cfg)
            self.assertTrue(run_dir.name.endswith("_ss4_n4_s1"))
            final_path = run_dir / "checkpoints" / "final.pt"

            checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
            self.assertIn("observation_encoder.image_encoders.0.head.0.temperature", checkpoint["model"])
            policy = DiffusionPolicy.from_checkpoint(checkpoint, "cpu").eval()
            self.assertIsInstance(policy.observation_encoder.image_encoders[0].head[0], SpatialSoftmax)
            rgb = torch.randint(0, 256, (2, policy.obs_horizon, 3, 64, 64), dtype=torch.uint8)
            proprio = torch.randn(2, policy.obs_horizon, 5)
            actions = policy.get_action(rgb, proprio, generator=torch.Generator().manual_seed(0))
            self.assertEqual(actions.shape, (2, policy.act_horizon, 4))
            self.assertTrue(torch.isfinite(actions).all())

            final_path.unlink()
            self.assertEqual(run_training(cfg, output_root=output_root, device="cpu"), 0)
            self.assertEqual(torch.load(final_path, map_location="cpu", weights_only=False)["step"], 2)


if __name__ == "__main__":
    unittest.main()
