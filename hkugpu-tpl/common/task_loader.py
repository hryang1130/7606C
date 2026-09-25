from __future__ import annotations

import importlib
import re


def load_task(name: str):
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
        raise ValueError("Task names may contain only letters, numbers, and underscores")
    try:
        task = importlib.import_module(f"tasks.{name}")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(f"Task plugin 'tasks/{name}' is missing or has an import error") from error
    required = ("config", "data", "models", "training", "policy", "environment")
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise AttributeError(f"Task plugin {name!r} is missing modules: {', '.join(missing)}")
    return task
