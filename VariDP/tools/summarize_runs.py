#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_runs.py — 把某个 runs/ 下的所有实验产物汇总成一份 Markdown 报告（机器无关）。

本机与学院 GPU 的结果都能汇总：`--runs-dir` 指到哪套就扫哪套
（`train_local/runs` 或 `train/runs`），也可以先把农场结果拷回来再一起扫。

扫描内容（每个实验目录）：
  train_summary.json   训练配置 + 指标（epochs / total_iters / seconds / best_score / n_params ...）
  eval_*.json          评测成功率 + 每个 episode 明细（成功与否 / 步数）

产出（默认写在 `<runs-dir>/` 下）：
  SUMMARY_all.md   可直接贴进报告的汇总表（`--out` 可改路径）
  SUMMARY_all.csv  同样的表（`--csv` 时额外导出，方便画图）

用法：
  python tools/summarize_runs.py                              # 默认扫 train_local/runs
  python tools/summarize_runs.py --env-id PickCube-v1
  python tools/summarize_runs.py --runs-dir train/runs        # 扫学院 GPU 那套结果
  python tools/summarize_runs.py --out runs/REPORT.md --csv

目录名约定（两种都支持）：
  <env>_frac<f>_seed<s>            （旧版，主干按 mlp 处理）
  <env>_frac<f>_<backbone>_seed<s> （带主干；cfg 里有 backbone 时以 cfg 为准）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import os.path as osp
import re
import sys
from datetime import datetime

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.abspath(osp.join(HERE, ".."))
DEFAULT_RUNS_DIR = osp.join(ROOT, "train_local", "runs")

BACKBONE_ORDER = {"mlp": 0, "unet": 1, "transformer": 2}
DIR_RE = re.compile(
    r"^(?P<env>.+)_frac(?P<frac>[\d.]+)"
    r"(?:_(?P<backbone>mlp|unet|transformer))?"
    r"_seed(?P<seed>\d+)$"
)
EVAL_RE = re.compile(r"^eval_seed(\d+)_n(\d+)(_raw)?(_steps(\d+))?\.json$")

DASH = "—"


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 读取失败 {path}: {e}")
        return None


def parse_dirname(name):
    m = DIR_RE.match(name)
    if not m:
        return {}
    return dict(env_id=m.group("env"), demo_frac=float(m.group("frac")),
                backbone=m.group("backbone") or "mlp", seed=int(m.group("seed")))


def parse_eval_name(fname):
    m = EVAL_RE.match(fname)
    if not m:
        return {}
    return dict(seed0=int(m.group(1)), n=int(m.group(2)),
                use_raw=bool(m.group(3)),
                inference_steps=int(m.group(5)) if m.group(5) else None)


def scan(runs_dir):
    """扫描 runs 目录，返回实验信息列表。"""
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        d = osp.join(runs_dir, name)
        if not osp.isdir(d):
            continue
        summ_path = osp.join(d, "train_summary.json")
        eval_paths = sorted(glob.glob(osp.join(d, "eval_*.json")))
        if not osp.exists(summ_path) and not eval_paths:
            continue  # 不是实验目录，跳过

        info = dict(exp=name, dir=d)
        info.update(parse_dirname(name))

        s = load_json(summ_path) if osp.exists(summ_path) else None
        cfg = (s or {}).get("cfg", {}) or {}
        info.update(dict(
            env_id=cfg.get("env_id", info.get("env_id")),
            demo_frac=cfg.get("demo_frac", info.get("demo_frac")),
            seed=cfg.get("seed", info.get("seed")),
            backbone=cfg.get("backbone", info.get("backbone", "mlp")),
            n_train_episodes=cfg.get("n_train_episodes"),
            n_val_episodes=cfg.get("n_val_episodes"),
            n_params=cfg.get("n_params"),
            control_mode=cfg.get("control_mode"),
            obs_mode=cfg.get("obs_mode"),
            sim_backend=cfg.get("sim_backend"),
            max_episode_steps=cfg.get("max_episode_steps") or 100,
            num_inference_timesteps=cfg.get("num_inference_timesteps"),
            model_kwargs=cfg.get("model_kwargs") or {},
            epochs=(s or {}).get("epochs"),
            total_iters=(s or {}).get("total_iters"),
            batch=(s or {}).get("batch"),
            seconds=(s or {}).get("seconds"),
            best_score=(s or {}).get("best_score"),
            final_train_loss=(s or {}).get("final_train_loss"),
            has_summary=s is not None,
            name_ok=bool(DIR_RE.match(name)),
        ))
        it, sec = info.get("total_iters"), info.get("seconds")
        info["iters_per_s"] = round(it / sec, 1) if (it and sec) else None

        # ---- 评测 ----
        info["evals"] = []
        for ep in eval_paths:
            ev = load_json(ep)
            if not ev:
                continue
            tag = parse_eval_name(osp.basename(ep))
            results = ev.get("results", []) or []
            n = ev.get("n_episodes", len(results))
            n_ok = ev.get("n_success", sum(1 for r in results if r.get("success")))
            fails = [r for r in results if not r.get("success")]
            n_steps = info["max_episode_steps"]
            timeouts = sum(1 for r in fails if r.get("steps", 0) >= n_steps)
            ok_steps = [r["steps"] for r in results if r.get("success")]
            info["evals"].append(dict(
                file=osp.basename(ep),
                seed0=ev.get("seed0", tag.get("seed0")),
                n_episodes=n, n_success=n_ok,
                success_rate=ev.get("success_rate", n_ok / max(1, n)),
                use_raw=ev.get("use_raw", tag.get("use_raw", False)),
                inference_steps=tag.get("inference_steps"),
                avg_steps=(round(sum(r.get("steps", 0) for r in results) / len(results), 1)
                           if results else None),
                avg_ok_steps=(round(sum(ok_steps) / len(ok_steps), 1) if ok_steps else None),
                n_fail=len(fails), n_timeout=timeouts,
                n_early=len(fails) - timeouts,
                fail_seeds=[r.get("seed") for r in fails],
            ))
        runs.append(info)
    return runs


def primary_eval(evals):
    """挑一条“主评测”：优先非 raw、非 steps 覆盖的。"""
    if not evals:
        return None
    plain = [e for e in evals if not e["use_raw"] and e["inference_steps"] is None]
    return (plain or evals)[0]


# --------------------------------------------------------------------------- #
# 排序 / 格式化
# --------------------------------------------------------------------------- #
def sort_key(r):
    return (str(r.get("env_id") or ""),
            -(r.get("demo_frac") or 0.0),
            BACKBONE_ORDER.get(r.get("backbone") or "mlp", 9),
            r.get("seed") if r.get("seed") is not None else 0)


def f(v, spec):
    return DASH if v is None else format(v, spec)


def pct(v):
    return DASH if v is None else f"{v * 100:.1f}%"


def params_m(v):
    return DASH if v is None else f"{v / 1e6:.3f}"


OVERVIEW_COLS = ["实验", "任务", "主干", "demo_frac", "训练ep", "参数量(M)", "epochs",
                 "iters", "用时(s)", "iters/s", "best val", "train loss",
                 "评测", "成功率", "成功/总数", "平均步数"]


def overview_row(r):
    pe = primary_eval(r["evals"])
    ev_col = DASH
    if pe:
        bits = [f"{pe['n_episodes']}ep@{pe['seed0']}"]
        if pe["use_raw"]:
            bits.append("raw")
        if pe["inference_steps"] is not None:
            bits.append(f"{pe['inference_steps']}steps")
        ev_col = " ".join(bits)
    return [
        r["exp"],
        r.get("env_id") or DASH,
        r.get("backbone") or DASH,
        f(r.get("demo_frac"), "g"),
        f(r.get("n_train_episodes"), "d"),
        params_m(r.get("n_params")),
        f(r.get("epochs"), "d"),
        f(r.get("total_iters"), ","),
        f(r.get("seconds"), ".1f"),
        f(r.get("iters_per_s"), ".1f"),
        f(r.get("best_score"), ".5f"),
        f(r.get("final_train_loss"), ".5f"),
        ev_col,
        pct(pe["success_rate"]) if pe else DASH,
        f"{pe['n_success']}/{pe['n_episodes']}" if pe else DASH,
        f(pe["avg_steps"], ".1f") if pe else DASH,
    ]


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def display_path(path, base):
    """尽量给相对路径；跨盘符时退回绝对路径（Windows 上 relpath 会抛 ValueError）。"""
    try:
        return osp.relpath(path, base)
    except ValueError:
        return path


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def build_report(runs, runs_dir, args):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_eval = sum(len(r["evals"]) for r in runs)
    L = []

    # ---- 头部 ----
    L += ["# runs 训练/评测汇总报告", "",
          f"- 生成时间：{now}",
          f"- 扫描目录：`{display_path(runs_dir, HERE)}`",
          f"- 实验目录：**{len(runs)}** 个；评测文件：**{n_eval}** 份"]
    if args.env_id:
        L.append(f"- 过滤任务：`{args.env_id}`（用 `--env-id` 指定）")
    L.append("")

    if not runs:
        L += ["> 没有扫描到任何实验目录。", ""]
        return "\n".join(L) + "\n"

    # ---- §1 总览 ----
    L += ["## 1. 总览", ""]
    L += md_table(OVERVIEW_COLS, [overview_row(r) for r in runs])
    L.append("")

    # ---- §2 按任务分组 ----
    L += ["## 2. 按任务对比", ""]
    envs = []
    for r in runs:
        if r.get("env_id") not in envs:
            envs.append(r.get("env_id"))
    for env in envs:
        grp = [r for r in runs if r.get("env_id") == env]
        best = [r for r in grp if r["evals"]]
        L += [f"### {env or '(未识别任务)'}", ""]
        rows, hdr = [], ["主干", "demo_frac", "训练ep", "参数量(M)", "iters", "用时(s)",
                         "iters/s", "成功率", "成功/总数", "平均步数", "best val"]
        for r in grp:
            pe = primary_eval(r["evals"])
            rows.append([
                r.get("backbone") or DASH,
                f(r.get("demo_frac"), "g"),
                f(r.get("n_train_episodes"), "d"),
                params_m(r.get("n_params")),
                f(r.get("total_iters"), ","),
                f(r.get("seconds"), ".1f"),
                f(r.get("iters_per_s"), ".1f"),
                pct(pe["success_rate"]) if pe else DASH,
                f"{pe['n_success']}/{pe['n_episodes']}" if pe else DASH,
                f(pe["avg_steps"], ".1f") if pe else DASH,
                f(r.get("best_score"), ".5f"),
            ])
        L += md_table(hdr, rows)
        L.append("")
        if best:
            top = max(best, key=lambda r: primary_eval(r["evals"])["success_rate"])
            pe = primary_eval(top["evals"])
            note = (f"> 最佳：**{top.get('backbone')}** —— "
                    f"{pct(pe['success_rate'])}（{pe['n_success']}/{pe['n_episodes']}）")
            if top.get("total_iters") and top.get("seconds"):
                note += f"，{top['total_iters']:,} iters / {top['seconds']:.1f} s"
            n_backbones = {r.get("backbone") for r in grp}
            if len(n_backbones) > 1:
                note += f"；本任务已横向对比 {len(n_backbones)} 种主干"
            L += [note, ""]
        else:
            L += ["> 尚无评测结果（只找到训练产物）。", ""]

    # ---- §3 评测明细（含失败分类）----
    L += ["## 3. 评测明细（含失败分类）", "",
          "> 超时 = 跑满 `max_episode_steps` 仍未成功；提前终止 = 中途 `terminated`（通常是物体掉落）。", ""]
    rows = []
    for r in runs:
        for e in r["evals"]:
            rows.append([
                r["exp"],
                e["file"],
                f"{e['n_success']}/{e['n_episodes']}",
                pct(e["success_rate"]),
                f(e["avg_steps"], ".1f"),
                f(e["avg_ok_steps"], ".1f"),
                e["n_fail"], e["n_timeout"], e["n_early"],
                ", ".join(str(s) for s in e["fail_seeds"]) if e["fail_seeds"] else DASH,
            ])
    if rows:
        L += md_table(["实验", "评测文件", "成功/总数", "成功率", "平均步数", "成功步数",
                       "失败数", "超时", "提前终止", "失败 seeds"], rows)
    else:
        L += ["（没有评测结果）"]
    L.append("")

    # ---- §4 训练配置明细 ----
    L += ["## 4. 训练配置明细", ""]
    rows = []
    for r in runs:
        mk = r.get("model_kwargs") or {}
        rows.append([
            r["exp"],
            r.get("obs_mode") or DASH,
            r.get("control_mode") or DASH,
            r.get("sim_backend") or DASH,
            f"{mk.get('obs_horizon', DASH)}/{mk.get('pred_horizon', DASH)}/{mk.get('action_horizon', DASH)}",
            f(r.get("max_episode_steps"), "d"),
            f(r.get("num_inference_timesteps"), "d"),
            f(r.get("batch"), "d"),
            f(r.get("n_val_episodes"), "d"),
            f(r.get("seed"), "d"),
        ])
    L += md_table(["实验", "obs_mode", "control_mode", "backend", "To/Tp/Ta",
                   "max_steps", "DDIM步数", "batch", "val ep", "训练seed"], rows)
    L.append("")

    # ---- §5 完整性提示 ----
    L += ["## 5. 数据完整性提示", ""]
    warns = []
    for r in runs:
        if not r["has_summary"]:
            warns.append(f"- `{r['exp']}`：缺 `train_summary.json`（只有评测结果，无法给出参数量/用时）")
        elif not r["evals"]:
            warns.append(f"- `{r['exp']}`：缺评测 JSON（训练完成但没跑 `eval.py`）")
        elif not r["name_ok"] and not r.get("env_id"):
            warns.append(f"- `{r['exp']}`：目录名不符合约定，任务/主干/seed 可能识别不到")
    if warns:
        L += warns
    else:
        L += ["- 所有实验目录都同时具备训练与评测产物。"]
    L.append("")

    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def main():
    try:  # Windows 控制台编码兜底
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="汇总 runs/ 下的训练+评测产物为 Markdown 报告")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                    help="实验根目录（默认 <repo>/train_local/runs；农场结果用 `--runs-dir train/runs`）")
    ap.add_argument("--out", default=None, help="输出 .md 路径（默认 <runs-dir>/SUMMARY_all.md）")
    ap.add_argument("--env-id", default=None, help="只看某个任务，如 PickCube-v1")
    ap.add_argument("--csv", action="store_true", help="额外导出同名 .csv（总览表）")
    ap.add_argument("--quiet", action="store_true", help="不把报告打到终端")
    a = ap.parse_args()

    runs_dir = osp.abspath(a.runs_dir)
    if not osp.isdir(runs_dir):
        print(f"[error] 找不到目录：{runs_dir}")
        return 1

    runs = scan(runs_dir)
    if a.env_id:
        runs = [r for r in runs if r.get("env_id") == a.env_id]
    runs.sort(key=sort_key)

    report = build_report(runs, runs_dir, a)

    out = osp.abspath(a.out) if a.out else osp.join(runs_dir, "SUMMARY_all.md")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)

    if a.csv:
        csv_path = osp.splitext(out)[0] + ".csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(OVERVIEW_COLS)
            for r in runs:
                w.writerow(overview_row(r))
        print(f"[done] CSV 已保存 {csv_path}")

    if not a.quiet:
        print(report)
    print(f"[done] 报告已保存 {out}（{len(runs)} 个实验）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
