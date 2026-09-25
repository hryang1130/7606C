from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path):
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(json.loads(line))
    return values


def generate(task_name: str, runs_dir: Path, output_dir: Path):
    task_runs = runs_dir / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = defaultdict(list)
    evaluations = defaultdict(list)
    for path in task_runs.rglob("metrics.jsonl"):
        model_name = path.parents[1].name
        histories[model_name].append((path.parent.name, _read_jsonl(path)))
    for path in task_runs.rglob("evaluation_*.csv"):
        model_name, run_name = path.parents[1].name, path.parent.name
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows:
            rates = [row["success"].lower() == "true" for row in rows]
            evaluations[model_name].append({
                "run": run_name, "episodes": len(rows), "successes": int(sum(rates)),
                "success_rate": float(np.mean(rates)), "path": str(path),
            })

    model_names = sorted(set(histories) | set(evaluations))
    train_fig, train_ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for model_name in model_names:
        for run_name, records in histories[model_name]:
            train_rows = [(r["step"], r["train_loss"]) for r in records if r.get("train_loss") is not None]
            val_rows = [(r["step"], r["validation_loss"]) for r in records if r.get("validation_loss") is not None]
            if train_rows:
                train_ax.plot(*zip(*train_rows), alpha=0.5, linewidth=1, label=f"{model_name}/{run_name} train")
            if val_rows:
                train_ax.plot(*zip(*val_rows), linewidth=2, linestyle="--", label=f"{model_name}/{run_name} val")
    train_ax.set(title=f"{task_name}: training and validation loss", xlabel="Optimizer step", ylabel="Diffusion loss")
    train_ax.grid(alpha=0.25)
    if model_names:
        train_ax.legend(fontsize=7, ncol=2)
    else:
        train_ax.text(0.5, 0.5, "No training metrics found", ha="center", va="center", transform=train_ax.transAxes)
    training_plot = output_dir / "training_curves.png"
    train_fig.savefig(training_plot, dpi=160)
    plt.close(train_fig)

    means, deviations = [], []
    comparison_rows = []
    for model_name in model_names:
        runs = evaluations[model_name]
        rates = [run["success_rate"] for run in runs]
        means.append(float(np.mean(rates)) if rates else 0.0)
        deviations.append(float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0)
        comparison_rows.append({
            "model": model_name, "training_runs": len(runs),
            "mean_success_rate": means[-1] if rates else "",
            "std_success_rate": deviations[-1] if rates else "",
            "mean_evaluation_episodes": float(np.mean([run["episodes"] for run in runs])) if runs else "",
        })
    compare_fig, compare_ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if model_names and any(evaluations.values()):
        positions = np.arange(len(model_names))
        compare_ax.bar(positions, means, yerr=deviations, capsize=5, color="#2b8a78")
        compare_ax.set_xticks(positions, model_names)
        compare_ax.set_ylim(0, 1)
        compare_ax.set_ylabel("Task success rate (mean ± std across training seeds)")
        compare_ax.grid(axis="y", alpha=0.25)
        for position, mean in zip(positions, means):
            compare_ax.text(position, max(mean + 0.025, 0.025), f"{mean:.1%}", ha="center", va="bottom")
    else:
        compare_ax.text(0.5, 0.5, "No evaluation results found", ha="center", va="center", transform=compare_ax.transAxes)
    compare_ax.set_title(f"{task_name}: model comparison")
    comparison_plot = output_dir / "model_comparison.png"
    compare_fig.savefig(comparison_plot, dpi=160)
    plt.close(compare_fig)

    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "training_runs", "mean_success_rate", "std_success_rate", "mean_evaluation_episodes"))
        writer.writeheader()
        writer.writerows(comparison_rows)
    summary = {
        model: {"training_runs": len(evaluations[model]), "per_seed": evaluations[model],
                "mean_success_rate": comparison_rows[index]["mean_success_rate"],
                "std_success_rate": comparison_rows[index]["std_success_rate"]}
        for index, model in enumerate(model_names)
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def embedded(path):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    table = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in row) + "</tr>"
        for row in comparison_rows
    )
    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(task_name)} experiment report</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 18px;color:#202a2a}}h1,h2{{color:#174e48}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ccd5d2;padding:9px;text-align:left}}th{{background:#eaf1ef}}section{{margin:28px 0}}</style>
<h1>{html.escape(task_name)} experiment report</h1>
<p>Generated from {html.escape(str(task_runs))}. Task success is evaluated in the simulator; losses are training diagnostics.</p>
<section><h2>Training curves</h2><img alt="Training and validation loss" src="{embedded(training_plot)}"></section>
<section><h2>Architecture comparison</h2><img alt="Success-rate comparison" src="{embedded(comparison_plot)}">
<table><thead><tr><th>Model</th><th>Evaluated training runs</th><th>Mean success rate</th><th>Std across runs</th><th>Episodes per run</th></tr></thead><tbody>{table}</tbody></table></section>
</html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(page, encoding="utf-8")
    print(f"HTML report: {report_path}")
    print(f"Training curves image: {training_plot}")
    print(f"Model comparison image: {comparison_plot}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate plots and a self-contained HTML training report")
    parser.add_argument(
        "--task",
        required=True,
        help="Task plugin name under tasks/ (for example, task_07)",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    generate(args.task, args.runs_dir, args.output_dir or Path("reports") / args.task)


if __name__ == "__main__":
    main()
