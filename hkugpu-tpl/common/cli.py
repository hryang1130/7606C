from __future__ import annotations

import argparse
from pathlib import Path

from common import evaluate as evaluation
from common import engine, report, smoke, video
from common.task_loader import load_task


def _task_parser(subparsers, command):
    parser = subparsers.add_parser(command)
    parser.add_argument(
        "--task",
        required=True,
        help="Task plugin name under tasks/ (for example, task_07)",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(description="Reusable ManiSkill experiment workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _task_parser(subparsers, "models")
    _task_parser(subparsers, "smoke")
    prepare = _task_parser(subparsers, "prepare-data")
    prepare.add_argument("--data-dir", type=Path, default=None)
    prepare.add_argument("--conversion-workers", type=int, default=8)

    train_parser = _task_parser(subparsers, "train")
    train_parser.add_argument("--model", default=None)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--steps", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--dataset", type=Path, default=None)
    train_parser.add_argument("--output", type=Path, default=Path("runs"))
    train_parser.add_argument("--num-demos", type=int, default=None)
    train_parser.add_argument("--validation-fraction", type=float, default=None)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--validate-every", type=int, default=1000)
    train_parser.add_argument("--save-every", type=int, default=5000)
    train_parser.add_argument("--no-compile", action="store_true")
    train_parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default="auto")

    evaluate_parser = _task_parser(subparsers, "evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--episodes", type=int, default=None)
    evaluate_parser.add_argument("--first-seed", type=int, default=None)
    evaluate_parser.add_argument("--num-envs", type=int, default=None)
    evaluate_parser.add_argument("--inference-steps", type=int, default=None)
    evaluate_parser.add_argument("--backend", default=None)
    evaluate_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser.add_argument("--no-compile", action="store_true")

    video_parser = _task_parser(subparsers, "video")
    video_parser.add_argument("--checkpoint", type=Path, required=True)
    video_parser.add_argument("--episodes", type=int, default=3)
    video_parser.add_argument("--seed", type=int, default=10000)
    video_parser.add_argument("--inference-steps", type=int, default=None)
    video_parser.add_argument("--output", type=Path, default=None)
    video_parser.add_argument("--no-compile", action="store_true")

    report_parser = _task_parser(subparsers, "report")
    report_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    report_parser.add_argument("--output-dir", type=Path, default=None)
    compare_parser = _task_parser(subparsers, "compare")
    compare_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    compare_parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()
    task = load_task(args.task)
    config = task.config.TASK_CONFIG
    if args.command == "models":
        print("\n".join(config["default_models"]))
    elif args.command == "smoke":
        smoke.smoke(args.task, task)
    elif args.command == "prepare-data":
        if not hasattr(task, "prepare_data"):
            raise RuntimeError(f"tasks/{args.task} does not provide prepare_data.prepare")
        task.prepare_data.prepare(args.data_dir or Path(config.get("data_dir", Path(config["dataset"]).parent)),
                                  args.conversion_workers)
    elif args.command == "train":
        args.model = args.model or config["default_models"][0]
        args.steps = config["default_steps"] if args.steps is None else args.steps
        args.batch_size = config["default_batch_size"] if args.batch_size is None else args.batch_size
        args.learning_rate = config["default_learning_rate"] if args.learning_rate is None else args.learning_rate
        args.validation_fraction = config["validation_fraction"] if args.validation_fraction is None else args.validation_fraction
        if args.steps < 1 or args.batch_size < 1 or args.log_every < 1 or args.validate_every < 1:
            raise ValueError("steps, batch-size, log-every, and validate-every must be positive")
        engine.train(args.task, task, args)
    elif args.command == "evaluate":
        args.episodes = config["evaluation_episodes"] if args.episodes is None else args.episodes
        args.first_seed = config["evaluation_first_seed"] if args.first_seed is None else args.first_seed
        args.num_envs = config["evaluation_envs"] if args.num_envs is None else args.num_envs
        args.inference_steps = config["inference_steps"] if args.inference_steps is None else args.inference_steps
        args.backend = args.backend or config["eval_backend"]
        if args.episodes < 1 or args.num_envs < 1:
            raise ValueError("episodes and num-envs must be positive")
        evaluation.evaluate(args.task, task, args)
    elif args.command == "video":
        args.inference_steps = args.inference_steps or config["inference_steps"]
        video.record_video(args.task, task, args)
    elif args.command in ("report", "compare"):
        output_dir = args.output_dir or Path("reports") / args.task
        report.generate(args.task, args.runs_dir, output_dir)


if __name__ == "__main__":
    main()
