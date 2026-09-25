#!/usr/bin/env bash
# Mac 端多设备同步脚本。所有操作都由 Mac 发起，远端从不主动连 Mac。
#
#   scripts/sync.sh push           Mac main → ubuntu / wsl（要求 Mac 工作区干净，不强推）
#   scripts/sync.sh fetch          取回远端提交，能 fast-forward 才合入 Mac
#   scripts/sync.sh pull-results   wsl results/ checkpoints/ logs/ → Mac（不删除 Mac 已有文件）
#   scripts/sync.sh data           ubuntu demos-*/ → Mac，Mac data/ → wsl，传完按清单校验
#   scripts/sync.sh manifest       在远端重新生成 manifests/*.sha256（新数据产生后执行，再提交）
#   scripts/sync.sh status         三端 HEAD、工作区与数据清单对照
#
# 远端必须保留 receive.denyCurrentBranch=updateInstead；push 被拒说明远端有未提交改动，
# 应先在远端处理（提交后用 fetch 取回，或丢弃），不要绕过。
set -euo pipefail

REMOTES="ubuntu wsl"
BRANCH=main

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

remote_dir() {
  case "$1" in
    ubuntu) echo "Coding/dp-manip" ;;
    wsl) echo "projects/dp-manip" ;;
    *) echo "unknown remote: $1" >&2; exit 2 ;;
  esac
}

# 每台远端对应的数据清单（路径相对该远端项目根）
remote_manifest() {
  case "$1" in
    ubuntu) echo "manifests/ubuntu-demos.sha256" ;;
    wsl) echo "manifests/wsl-data.sha256" ;;
  esac
}

remote_data_glob() {
  case "$1" in
    ubuntu) echo "demos-*" ;;
    wsl) echo "data" ;;
  esac
}

on() { local host=$1; shift; ssh -o ConnectTimeout=5 "$host" "cd ~/$(remote_dir "$host") && $*"; }

require_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "Mac 工作区有未提交改动，先提交或还原：" >&2
    git status --short >&2
    exit 1
  fi
}

cmd_push() {
  require_clean
  local failed=0
  for r in $REMOTES; do
    echo "== push → $r"
    if ! git push "$r" "$BRANCH"; then
      echo "!! $r 拒绝 push（远端有未提交改动或不是 fast-forward），未强推" >&2
      failed=1
    fi
  done
  return $failed
}

cmd_fetch() {
  local failed=0
  for r in $REMOTES; do
    echo "== fetch ← $r"
    git fetch -q "$r"
    local theirs="$r/$BRANCH"
    if git merge-base --is-ancestor "$theirs" HEAD; then
      echo "   $r 没有 Mac 缺少的提交"
    elif git merge-base --is-ancestor HEAD "$theirs"; then
      require_clean
      echo "   fast-forward 到 $theirs："
      git log --oneline "HEAD..$theirs"
      git merge -q --ff-only "$theirs"
    else
      echo "!! Mac 与 $theirs 已分叉，需人工处理（git log --graph HEAD $theirs）" >&2
      failed=1
    fi
  done
  return $failed
}

cmd_pull_results() {
  local dir; dir=$(remote_dir wsl)
  for d in results checkpoints logs; do
    echo "== rsync wsl:$dir/$d/ → $d/"
    mkdir -p "$d"
    rsync -a "wsl:$dir/$d/" "$d/"
  done
}

cmd_data() {
  echo "== rsync ubuntu demos-*/ → Mac"
  rsync -a "ubuntu:$(remote_dir ubuntu)/demos-*" ./
  shasum -a 256 -c --quiet "$(remote_manifest ubuntu)" && echo "   Mac 副本与 ubuntu 清单一致"

  echo "== rsync Mac data/ → wsl"
  shasum -a 256 -c --quiet "$(remote_manifest wsl)" || { echo "!! Mac data/ 与清单不符，停止" >&2; exit 1; }
  rsync -a data/ "wsl:$(remote_dir wsl)/data/"
  on wsl "sha256sum -c --quiet -" < "$(remote_manifest wsl)" && echo "   wsl data/ 与清单一致"
}

cmd_manifest() {
  for r in $REMOTES; do
    local m; m=$(remote_manifest "$r")
    echo "== 生成 $m（在 $r 上）"
    on "$r" "find $(remote_data_glob "$r") -type f | LC_ALL=C sort | xargs sha256sum" > "$m"
    wc -l < "$m" | sed 's/^/   文件数：/'
  done
  git status --short manifests/
  echo "确认变化后提交 manifests/。"
}

cmd_status() {
  echo "== Mac  $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"
  git status --short
  for m in manifests/*.sha256; do
    if shasum -a 256 -c --quiet "$m" 2>/dev/null; then echo "   数据 $m：OK"; else echo "   数据 $m：不一致或缺失"; fi
  done
  for r in $REMOTES; do
    echo "== $r"
    on "$r" "echo \"   HEAD \$(git rev-parse --short HEAD)\"; git status --short" || { echo "!! 无法连接 $r" >&2; continue; }
    if on "$r" "sha256sum -c --quiet -" < "$(remote_manifest "$r")" >/dev/null 2>&1; then
      echo "   数据 $(remote_manifest "$r")：OK"
    else
      echo "   数据 $(remote_manifest "$r")：不一致或缺失"
    fi
  done
}

case "${1:-}" in
  push) cmd_push ;;
  fetch) cmd_fetch ;;
  pull-results) cmd_pull_results ;;
  data) cmd_data ;;
  manifest) cmd_manifest ;;
  status) cmd_status ;;
  *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
