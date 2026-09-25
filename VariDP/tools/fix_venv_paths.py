"""改目录名/搬家后，修复 venv 里写死的绝对路径。

背景：`python -m venv` 建出来的环境会把**绝对路径**写进几个文件里，项目目录一改名就失效：

  venv/Scripts/activate.bat / Activate.ps1    VIRTUAL_ENV 指向旧路径（命令行 activate 会指错）
  venv/pyvenv.cfg                             记录创建时的 command 行
  venv/Lib/site-packages/_pinocchio_dlls.pth  本项目给 pinocchio DLL 加的加载路径（失效后
                                              control-mode 转换 / replay 会报 PinocchioModel 错误）

直接运行 `venv\\Scripts\\python.exe`、以及 PyCharm 里选解释器**都不受**这些影响；
只有用到 `activate` 或 replay 转换时才需要修。

用法（在仓库根目录用任意 python 跑，不必先激活 venv）：
  python tools/fix_venv_paths.py            # 只预览要改什么（默认 dry-run）
  python tools/fix_venv_paths.py --apply    # 真正改写
"""
from __future__ import annotations

import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VENV_DIRS = ["venv", ".venv"]

# 各平台布局都可能出现，存在才处理
REL_FILES = [
    os.path.join("Scripts", "activate.bat"),
    os.path.join("Scripts", "Activate.ps1"),
    "pyvenv.cfg",
    os.path.join("Lib", "site-packages", "_pinocchio_dlls.pth"),
    os.path.join("lib", "site-packages", "_pinocchio_dlls.pth"),
    os.path.join("bin", "activate"),
]

# 只认"赋值/字符串里"的 venv 路径：前面必须是 = ' " ( 或行首，
# 这样不会误改帮助文档里的示例（如 Activate.ps1 的 "Activate.ps1 -VenvDir C:\Users\MyUser\Common\.venv"）。
# 路径含空格的项目（如 "D:\My Projects\..."）本脚本不处理，需手动改——避免误吃整行命令。
HEAD = r"(?:(?<=^)|(?<=[='\"(]))"
TAIL = r"(?=[\\/'\"\s)]|$)"
WIN_RE = re.compile(HEAD + r"[A-Za-z]:[\\/][^'\"\r\n\s]*?[\\/](?:\.venv|venv)" + TAIL)
POSIX_RE = re.compile(HEAD + r"/(?:[^'\"\r\n\s]*?)/(?:\.venv|venv)" + TAIL)

# pyvenv.cfg 的 command 行形如 "command = C:\...\python.exe -m venv D:\...\venv"，
# 上面那条正则会把整行吃掉，所以单独处理：只替换 "-m venv" 后面的那个路径。
VENV_CMD_RE = re.compile(r"(-m venv\s+)([^\s]+)")


def fix_text(text: str, new_venv: str, is_pyvenv_cfg: bool = False) -> "tuple[str, int]":
    """把文本里指向旧 venv 的绝对路径替换成 new_venv，返回 (新文本, 替换次数)。"""
    n = 0

    def repl(_m):
        nonlocal n
        n += 1
        return new_venv

    if is_pyvenv_cfg:
        def repl_cmd(m):
            nonlocal n
            n += 1
            return m.group(1) + new_venv

        return VENV_CMD_RE.sub(repl_cmd, text), n

    out = WIN_RE.sub(repl, text)
    out = POSIX_RE.sub(repl, out)
    return out, n


def main() -> int:
    ap = argparse.ArgumentParser(description="修复 venv 里写死的绝对路径（搬家/改名后跑）")
    ap.add_argument("--apply", action="store_true", help="真正写回文件（默认只预览）")
    a = ap.parse_args()

    print(f"[repo] {ROOT}")
    changed_total = 0
    seen = set()
    for vname in VENV_DIRS:
        vdir = os.path.join(ROOT, vname)
        if not os.path.isdir(vdir):
            continue
        for rel in REL_FILES:
            fp = os.path.join(vdir, rel)
            if not os.path.isfile(fp) or os.path.realpath(fp) in seen:
                continue
            seen.add(os.path.realpath(fp))     # 大小写不敏感的文件系统上 lib/Lib 是同一个文件
            try:
                text = open(fp, encoding="utf-8", errors="surrogateescape").read()
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {vname}/{rel}: 读不了（{e}）")
                continue
            if rel == "pyvenv.cfg":
                new_text, n = fix_text(text, vdir, is_pyvenv_cfg=True)
            else:
                # 逐行处理，跳过注释行（Activate.ps1 的注释里带了 "C:\Users\MyUser\Common\.venv" 这样的示例）
                out_lines, n = [], 0
                for line in text.splitlines(keepends=True):
                    if line.lstrip().startswith("#"):
                        out_lines.append(line)
                        continue
                    fixed, k = fix_text(line, vdir)
                    out_lines.append(fixed)
                    n += k
                new_text = "".join(out_lines)
            if n == 0 or new_text == text:
                print(f"  [ok]   {vname}/{rel}: 已是当前路径，无需修改")
                continue
            changed_total += n
            print(f"  [{'fix' if a.apply else 'dry'}] {vname}/{rel}: {n} 处 -> {vdir}")
            if a.apply and new_text != text:
                with open(fp, "w", encoding="utf-8", errors="surrogateescape",
                          newline="") as fh:
                    fh.write(new_text)

    if changed_total == 0:
        print("[done] venv 里的路径都是当前目录，不需要修。")
    elif a.apply:
        print(f"[done] 已修 {changed_total} 处。")
    else:
        print(f"[dry-run] 共需修 {changed_total} 处；加 --apply 才真正写入。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
