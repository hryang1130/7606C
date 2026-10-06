"""Small, explicit helpers for the separately recorded second study."""
from __future__ import annotations

import json
from pathlib import Path


def paired_changes(left: dict, right: dict) -> dict:
    """Compare matched seeds; fail on duplicated or missing episodes."""
    def indexed(result):
        episodes = result["episodes"]
        values = {int(e["seed"]): bool(e["success_once"]) for e in episodes}
        if len(values) != len(episodes):
            raise ValueError("duplicated episode seeds")
        return values

    a, b = indexed(left), indexed(right)
    if not a or a.keys() != b.keys():
        raise ValueError("paired results must contain the same nonempty seed set")
    gained = sorted(s for s in a if a[s] and not b[s])
    lost = sorted(s for s in a if not a[s] and b[s])
    return {"gained": gained, "lost": lost, "pairs": len(a),
            "difference": (len(gained) - len(lost)) / len(a)}


def choose_candidate(rows: list[dict], tie_episodes: int = 2) -> dict:
    """Within two successes of best, prefer smaller alpha then fewer steps."""
    best = max(row["successes"] for row in rows)
    eligible = [r for r in rows if r["successes"] >= best - tie_episodes]
    return min(eligible, key=lambda r: (r["alpha"], r["steps"]))


def write_once(path: Path, payload: dict) -> dict:
    """Persist immutable study decisions and reject a changed rerun."""
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != payload:
            raise ValueError(f"immutable decision differs: {path}")
        return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return payload
