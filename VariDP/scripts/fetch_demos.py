"""fetch_demos.py — 从 hf-mirror 镜像下载 ManiSkill 官方演示并解包到 <demo_dir>/<Task>-v1/。

为什么要它：`mani_skill.utils.download_demo` 直连 huggingface.co（本机 502），
而 hf-mirror 不带 User-Agent 会返回 403 —— 这里两件事都处理好了。

用法（在仓库根目录，用 venv 的 python；需要联网）：
  python scripts/fetch_demos.py PickCube-v1 [StackCube-v1 ...]
  python scripts/fetch_demos.py --out D:\\demos PickCube-v1     # 默认 mani_skill.DEMO_DIR
  python scripts/fetch_demos.py --list                         # 只列出可选任务

下载完的原始演示是 `pd_joint_pos` 动作空间，还需 `scripts/convert_all.py` 重放转换成
`state` + `pd_ee_delta_pos`（训练用）。
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

try:
    from mani_skill import DEMO_DIR
except Exception:                       # 没装 mani_skill 也能用来下载
    DEMO_DIR = os.path.expanduser("~/.maniskill/demos")

BASE = "https://hf-mirror.com/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos"
HDR = {"User-Agent": "Mozilla/5.0"}   # 关键：hf-mirror 不带 UA 会返回 403

# 有官方演示的任务（见 ManiSkill_Demonstrations 数据集）
AVAILABLE = [
    "PickCube-v1", "StackCube-v1", "PushCube-v1", "PullCube-v1",
    "PegInsertionSide-v1", "PlugCharger-v1", "PlaceSphere-v1",
    "LiftPegUpright-v1", "PullCubeTool-v1", "DrawTriangle-v1", "DrawSVG-v1",
    "StackPyramid-v1", "PokeCube-v1", "RollBall-v1", "PushT-v1",
    "AnymalC-Reach-v1", "TwoRobotPickCube-v1", "TwoRobotStackCube-v1",
]


def fetch(env_id: str, out_dir: str = None):
    out_dir = out_dir or DEMO_DIR
    os.makedirs(out_dir, exist_ok=True)
    url = f"{BASE}/{env_id}.zip?download=true"
    print("GET", url, flush=True)
    req = urllib.request.Request(url, headers=HDR)
    data = urllib.request.urlopen(req, timeout=1800).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    z.extractall(out_dir)          # zip 内部已含 <env_id>/ 前缀
    print(f"{env_id}: {len(data)/1e6:.1f} MB 已解包到 {out_dir}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 ManiSkill 官方演示（hf-mirror 镜像）")
    ap.add_argument("env_ids", nargs="*", help="任务 id，如 PickCube-v1；缺省 PickCube-v1")
    ap.add_argument("--out", default=str(DEMO_DIR), help="解包目录（默认 mani_skill.DEMO_DIR）")
    ap.add_argument("--list", action="store_true", help="只列出可选任务，不下载")
    a = ap.parse_args()

    if a.list:
        print("\n".join(AVAILABLE))
        return 0
    for eid in (a.env_ids or ["PickCube-v1"]):
        fetch(eid, a.out)
    print("\n下一步：python scripts/convert_all.py --tasks " +
          " ".join(a.env_ids or ["PickCube-v1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
