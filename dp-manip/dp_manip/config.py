"""Layered, typed experiment configuration for training and evaluation."""

from __future__ import annotations

import copy
import dataclasses
import json
try:
    import tomllib                    # Python 3.11+
except ModuleNotFoundError:           # Python 3.10：tomli 是官方 backport
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar


@dataclass
class TaskConfig:
    name: str
    env_id: str
    control_mode: str
    max_episode_steps: int
    sim_backend: str
    shader_pack: str


@dataclass
class DataConfig:
    root: str
    train_path: str
    val_path: str
    num_demos: int
    val_num_demos: int
    # Decode the selected episodes into RAM before training instead of
    # decompressing gzip chunks per window. Runtime only: the samples are
    # bit-identical either way, so it is not part of the control hash.
    preload: bool


# Pooling heads of ``dp_manip.vision.ResNet18Encoder``.
VISION_POOLS = ("avg", "spatial_softmax")


@dataclass
class VisionConfig:
    feature_dim: int
    random_shift: int
    share_camera_encoder: bool
    # How the final ResNet feature map becomes ``feature_dim`` features:
    # "avg" (global average pooling) or "spatial_softmax" (robomimic keypoints).
    pool: str
    # Keypoints of the spatial softmax; unused by "avg".
    num_keypoints: int


@dataclass
class PolicyConfig:
    obs_horizon: int
    act_horizon: int
    pred_horizon: int
    diffusion_step_embed_dim: int
    unet_dims: list[int]
    kernel_size: int
    n_groups: int
    # Structural selector, not a scientific hyperparameter.
    backbone: str
    # Backbone architecture definitions, not scientific hyperparameters. Every
    # arm resolves the same baseline values and ``backbone`` selects which group
    # is consumed, so a backbone comparison changes only the selector. Recorded
    # configs that predate a field get it from ``_HISTORICAL_VALUES``.
    transformer_layers: int
    transformer_heads: int
    transformer_embed_dim: int
    transformer_dropout_emb: float
    transformer_dropout_attn: float
    transformer_causal_attn: bool
    transformer_cond_layers: int
    mlp_hidden_dim: int
    mlp_layers: int
    mlp_time_embed_dim: int
    mlp_obs_feat_dim: int


@dataclass
class TrainConfig:
    seed: int
    total_iters: int
    batch_size: int
    num_workers: int
    lr: float
    betas: list[float]
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    log_freq: int
    resume_freq: int
    validation_steps: list[int]
    checkpoint_steps: list[int]
    amp: bool


@dataclass
class EmaConfig:
    decay: float


@dataclass
class DiffusionConfig:
    num_diffusion_iters: int
    num_inference_iters: int


@dataclass
class EvalConfig:
    val_seed_start: int
    val_episodes: int
    test_seed_start: int
    test_episodes: int
    num_envs: int
    inference_seed: int


@dataclass
class Config:
    task: TaskConfig
    data: DataConfig
    vision: VisionConfig
    policy: PolicyConfig
    train: TrainConfig
    ema: EmaConfig
    diffusion: DiffusionConfig
    eval: EvalConfig

    def validate(self) -> None:
        if self.task.sim_backend != "physx_cpu":
            raise ValueError("evaluation must use physx_cpu to match the generated demonstrations")
        if (
            isinstance(self.data.num_demos, bool)
            or not isinstance(self.data.num_demos, int)
            or self.data.num_demos < 1
        ):
            raise ValueError("data.num_demos must be positive")
        if (
            isinstance(self.data.val_num_demos, bool)
            or not isinstance(self.data.val_num_demos, int)
            or self.data.val_num_demos < 1
        ):
            raise ValueError("data.val_num_demos must be positive")
        if not isinstance(self.data.preload, bool):
            raise ValueError("data.preload must be a boolean")
        policy = self.policy
        if not isinstance(policy.backbone, str) or not policy.backbone:
            raise ValueError("policy.backbone must be a non-empty string")
        if policy.transformer_layers < 1 or policy.transformer_heads < 1:
            raise ValueError(
                "policy.transformer_layers and policy.transformer_heads must be positive"
            )
        if (
            policy.transformer_embed_dim < 2
            or policy.transformer_embed_dim % 2
            or policy.transformer_embed_dim % policy.transformer_heads
        ):
            raise ValueError(
                "policy.transformer_embed_dim must be an even positive multiple of "
                "policy.transformer_heads"
            )
        for name in ("transformer_dropout_emb", "transformer_dropout_attn"):
            if not 0.0 <= getattr(policy, name) < 1.0:
                raise ValueError(f"policy.{name} must be in [0, 1)")
        if policy.transformer_cond_layers < 0:
            raise ValueError("policy.transformer_cond_layers must be non-negative")
        if policy.mlp_hidden_dim < 1 or policy.mlp_layers < 1 or policy.mlp_obs_feat_dim < 1:
            raise ValueError(
                "policy.mlp_hidden_dim, policy.mlp_layers and policy.mlp_obs_feat_dim must be positive"
            )
        if policy.mlp_time_embed_dim < 2 or policy.mlp_time_embed_dim % 2:
            raise ValueError("policy.mlp_time_embed_dim must be an even positive number")
        if min(policy.obs_horizon, policy.act_horizon, policy.pred_horizon) < 1:
            raise ValueError("all horizons must be positive")
        if policy.obs_horizon + policy.act_horizon - 1 > policy.pred_horizon:
            raise ValueError("need obs_horizon + act_horizon - 1 <= pred_horizon")
        diffusion = self.diffusion
        if diffusion.num_diffusion_iters < 1 or diffusion.num_inference_iters < 1:
            raise ValueError("diffusion iteration counts must be positive")
        if diffusion.num_inference_iters > diffusion.num_diffusion_iters:
            raise ValueError("inference diffusion iterations cannot exceed training iterations")
        if self.vision.feature_dim < 1 or self.vision.random_shift < 0:
            raise ValueError("invalid vision encoder settings")
        if self.vision.pool not in VISION_POOLS:
            raise ValueError(
                f"vision.pool must be one of {list(VISION_POOLS)}, got {self.vision.pool!r}"
            )
        if (
            isinstance(self.vision.num_keypoints, bool)
            or not isinstance(self.vision.num_keypoints, int)
            or self.vision.num_keypoints < 1
        ):
            raise ValueError("vision.num_keypoints must be a positive integer")
        train = self.train
        if min(train.total_iters, train.batch_size, train.log_freq, train.resume_freq) < 1:
            raise ValueError("training counts must be positive")
        for step in [*train.validation_steps, *train.checkpoint_steps]:
            if step < 1:
                raise ValueError(f"intermediate step {step} must be positive")
        if train.lr <= 0.0:
            raise ValueError("train.lr must be positive")
        betas = train.betas
        if (
            not isinstance(betas, (list, tuple))
            or len(betas) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 1.0
                for value in betas
            )
        ):
            raise ValueError("train.betas must be two numbers in [0, 1)")
        if train.weight_decay < 0.0:
            raise ValueError("train.weight_decay must be non-negative")
        if train.grad_clip <= 0.0:
            raise ValueError("train.grad_clip must be positive")
        if train.num_workers < 0 or train.warmup_steps < 0:
            raise ValueError("train.num_workers and train.warmup_steps must be non-negative")
        if not 0.0 <= self.ema.decay < 1.0:
            raise ValueError("ema.decay must be in [0, 1)")
        evaluation = self.eval
        if evaluation.val_episodes < 1 or evaluation.test_episodes < 1 or evaluation.num_envs < 1:
            raise ValueError("evaluation counts must be positive")
        if min(evaluation.val_seed_start, evaluation.test_seed_start, evaluation.inference_seed) < 0:
            raise ValueError("evaluation seeds must be non-negative")
        if set(self.val_seeds()) & set(self.test_seeds()):
            raise ValueError("validation and test rollout seed ranges overlap")

    def val_seeds(self) -> list[int]:
        return list(range(self.eval.val_seed_start, self.eval.val_seed_start + self.eval.val_episodes))

    def test_seeds(self) -> list[int]:
        return list(range(self.eval.test_seed_start, self.eval.test_seed_start + self.eval.test_episodes))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def default_run_name(cfg: Config) -> str:
    """Return the run directory name shared by the trainer and the sweep.

    The backbone is part of the name so different backbone arms with the same
    task, data size, and seed never share a run directory. A non-default vision
    pool adds a tag (``_ss32`` for a 32-keypoint spatial softmax); average
    pooling adds none, so existing run directories keep their names and the
    avg arm of a pooling comparison reuses the matching baseline run.
    """
    pool = "" if cfg.vision.pool == "avg" else f"_ss{cfg.vision.num_keypoints}"
    return (
        f"{cfg.task.name}_rgb_{cfg.policy.backbone}{pool}"
        f"_n{cfg.data.num_demos}_s{cfg.train.seed}"
    )


@dataclass(frozen=True)
class ExperimentSpec:
    """A declared experiment variable, grid values, and replicate seeds."""

    name: str
    variable: str
    values: tuple[Any, ...]
    replicates: Mapping[str, tuple[int, ...]]
    # Closed-loop ``--split train`` diagnostic budget: the first K training
    # seeds in ascending order. ``None`` evaluates the whole training subset.
    train_eval_episodes: int | None = None
    # ``section.key -> value`` set in every cell (for example the data size a
    # model comparison runs at). Like the variable, runtime overrides cannot
    # change them, so every run of the grid trains at the declared values.
    fixed: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def seeds_for(self, value: Any) -> tuple[int, ...]:
        try:
            return self.replicates[str(value)]
        except KeyError as error:
            raise ValueError(f"experiment {self.name!r} has no replicates for value {value!r}") from error


_T = TypeVar("_T")
_SECTIONS = {
    "task": TaskConfig,
    "data": DataConfig,
    "vision": VisionConfig,
    "policy": PolicyConfig,
    "train": TrainConfig,
    "ema": EmaConfig,
    "diffusion": DiffusionConfig,
    "eval": EvalConfig,
}
# Everything else comes from baseline.toml, so tasks cannot drift from it.
_TASK_LAYER_KEYS = {
    "task": frozenset({"name", "env_id", "control_mode", "max_episode_steps"}),
    "data": frozenset({"train_path", "val_path"}),
}


def _section(cls: type[_T], raw: dict[str, Any], name: str) -> _T:
    values = raw.get(name, {})
    if not isinstance(values, dict):
        raise ValueError(f"config section {name!r} must be a table")
    known = {item.name for item in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in [{name}]: {sorted(unknown)}")
    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(f"invalid or incomplete [{name}] section: {error}") from error


def _adapt_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Map version-1 checkpoint config fields at the checkpoint boundary."""
    raw = copy.deepcopy(raw)
    policy = raw.get("policy")
    if isinstance(policy, dict):
        diffusion = raw.setdefault("diffusion", {})
        for name in ("num_diffusion_iters", "num_inference_iters"):
            if name in policy:
                value = policy.pop(name)
                if name in diffusion and diffusion[name] != value:
                    raise ValueError(f"conflicting legacy policy.{name} and diffusion.{name}")
                diffusion[name] = value
    train = raw.get("train")
    if isinstance(train, dict) and "ema_decay" in train:
        ema = raw.setdefault("ema", {})
        value = train.pop("ema_decay")
        if "decay" in ema and ema["decay"] != value:
            raise ValueError("conflicting legacy train.ema_decay and ema.decay")
        ema["decay"] = value
    return raw


# Values that recorded configs (checkpoints, resume.pt, run.json) written before
# a field existed actually trained with. They are applied only by
# ``from_recorded``; fresh configs must get every value from baseline.toml.
_HISTORICAL_VALUES: dict[tuple[str, str], Any] = {
    # Before Phase 7 the UNet was the only noise predictor.
    ("policy", "backbone"): "unet",
    # Before Phases 9-11 the backbone structure fields did not exist; they only
    # describe the non-UNet arms, whose donor structure these are.
    ("policy", "transformer_layers"): 8,
    ("policy", "transformer_heads"): 4,
    ("policy", "transformer_embed_dim"): 256,
    ("policy", "transformer_dropout_emb"): 0.0,
    ("policy", "transformer_dropout_attn"): 0.3,
    ("policy", "transformer_causal_attn"): True,
    ("policy", "transformer_cond_layers"): 0,
    ("policy", "mlp_hidden_dim"): 256,
    ("policy", "mlp_layers"): 3,
    ("policy", "mlp_time_embed_dim"): 128,
    ("policy", "mlp_obs_feat_dim"): 256,
    # Before data.preload existed every run read RGB windows lazily from HDF5.
    ("data", "preload"): False,
    # Before Phase 17 the trainer hard-coded AdamW betas (0.95, 0.999).
    ("train", "betas"): [0.95, 0.999],
    # Before vision.pool existed the encoder always average-pooled; the
    # keypoint count is inert under "avg" and matches today's baseline.
    ("vision", "pool"): "avg",
    ("vision", "num_keypoints"): 32,
}


def from_recorded(raw: dict[str, Any], overrides: Sequence[str] = ()) -> Config:
    """Rebuild the config of a recorded run (checkpoint, resume.pt or run.json).

    Unlike :func:`from_dict`, fields added after the run was recorded are filled
    with the values that run actually used, so older artifacts stay loadable.
    ``overrides`` (``section.key=value``) are applied afterwards; fine-tuning
    uses them to start from a baseline checkpoint's exact config.
    """
    raw = _adapt_legacy_config(raw)
    for (section, key), value in _HISTORICAL_VALUES.items():
        values = raw.get(section)
        if isinstance(values, dict) and key not in values:
            values[key] = copy.deepcopy(value)
    for item in overrides:
        key, value = _parse_override(item)
        _set_dotted(raw, key, value)
    return from_dict(raw)


# Keys that change how a run reads its data but never the samples it trains
# on. Runs recorded lazily (preload=false, the default before Phase 21) are the
# same run as a preloaded invocation, so they stay completed and resumable.
_EXECUTION_ONLY = (("data", "preload"),)


def same_run(recorded: Config, cfg: Config) -> bool:
    """Whether a recorded run (``run.json`` / ``resume.pt``) is ``cfg``'s run."""
    ours, theirs = recorded.to_dict(), cfg.to_dict()
    for section, key in _EXECUTION_ONLY:
        ours[section].pop(key, None)
        theirs[section].pop(key, None)
    return ours == theirs


def from_dict(raw: dict[str, Any]) -> Config:
    """Build a resolved config, adapting the version-1 checkpoint layout."""
    raw = _adapt_legacy_config(raw)
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    cfg = Config(**{name: _section(cls, raw, name) for name, cls in _SECTIONS.items()})
    cfg.validate()
    return cfg


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def resolve_config_path(name: str | Path, directory: str | Path, kind: str) -> Path:
    """Resolve a short config name to ``<directory>/<name>.toml`` or an existing path.

    Entry points own the directories (``configs/tasks``, ``configs/experiments``);
    this helper only implements the name-or-path convention they share.
    """
    path = Path(name)
    if path.is_file():
        return path
    candidate = Path(directory) / f"{name}.toml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"unknown {kind} {name!r}; expected {candidate} or an existing path")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_dotted(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    section, dot, name = dotted_key.partition(".")
    if not dot or section not in _SECTIONS or not name or "." in name:
        raise ValueError(f"override must look like section.key: {dotted_key!r}")
    raw.setdefault(section, {})[name] = value


def _parse_override(item: str) -> tuple[str, Any]:
    key, separator, value = item.partition("=")
    if not separator:
        raise ValueError(f"override must look like section.key=value: {item!r}")
    try:
        parsed = tomllib.loads(f"value = {value}")["value"]
    except tomllib.TOMLDecodeError:
        parsed = value
    return key, parsed


def load_experiment(path: str | Path) -> ExperimentSpec:
    """Load a sweep definition without leaking its grid into core config code."""
    return _parse_experiment(_read_toml(Path(path)))


def load_experiment_optional(path: str | Path) -> ExperimentSpec | None:
    """Return the spec declared by a TOML file, or ``None`` for plain overrides."""
    raw = _read_toml(Path(path))
    return _parse_experiment(raw) if "experiment" in raw else None


def _parse_experiment(raw: dict[str, Any]) -> ExperimentSpec:
    unknown = set(raw) - {"experiment", "replicates", "diagnostics", "fixed"}
    if unknown:
        raise ValueError(f"unknown experiment sections: {sorted(unknown)}")
    experiment = raw.get("experiment", {})
    if set(experiment) != {"name", "variable", "values"}:
        raise ValueError("[experiment] must define exactly name, variable, and values")
    variable = experiment["variable"]
    values = experiment["values"]
    if not isinstance(variable, str) or not isinstance(values, list) or not values:
        raise ValueError("experiment variable must be a string and values must be a non-empty list")
    probe: dict[str, Any] = {}
    _set_dotted(probe, variable, values[0])
    raw_replicates = raw.get("replicates", {})
    if not isinstance(raw_replicates, dict):
        raise ValueError("[replicates] must be a table")
    replicates = {
        str(value): tuple(int(seed) for seed in raw_replicates.get(str(value), ()))
        for value in values
    }
    if any(not seeds or any(seed < 0 for seed in seeds) for seeds in replicates.values()):
        raise ValueError("every experiment value must define non-negative replicate seeds")
    extra = set(raw_replicates) - {str(value) for value in values}
    if extra:
        raise ValueError(f"replicates defined for unknown values: {sorted(extra)}")
    diagnostics = raw.get("diagnostics", {})
    if not isinstance(diagnostics, dict) or set(diagnostics) - {"train_eval_episodes"}:
        raise ValueError("[diagnostics] may only define train_eval_episodes")
    train_eval_episodes = diagnostics.get("train_eval_episodes")
    if train_eval_episodes is not None and (
        isinstance(train_eval_episodes, bool)
        or not isinstance(train_eval_episodes, int)
        or train_eval_episodes < 1
    ):
        raise ValueError("diagnostics.train_eval_episodes must be a positive integer")
    fixed = raw.get("fixed", {})
    if not isinstance(fixed, dict):
        raise ValueError("[fixed] must be a table of quoted section.key entries")
    for key in fixed:
        _set_dotted(probe, key, fixed[key])
        if key == variable:
            raise ValueError(f"[fixed] cannot set the experiment variable {variable!r}")
    return ExperimentSpec(
        str(experiment["name"]),
        variable,
        tuple(values),
        replicates,
        train_eval_episodes,
        copy.deepcopy(fixed),
    )


def match_experiment_value(spec: ExperimentSpec, value: Any) -> Any:
    """Return the declared grid value selected by a CLI string.

    ``--value`` arrives as text, so ``"50"`` must select the declared integer
    ``50`` and ``"transformer"`` the declared string. Matching on the string
    form keeps the data-size and backbone grids on one entry point.
    """
    for declared in spec.values:
        if declared == value or str(declared) == str(value):
            return declared
    raise ValueError(
        f"value {value!r} is not in experiment {spec.name!r}: {list(spec.values)}"
    )


def _resolve_legacy_task(path: Path, raw: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if set(raw) != {"legacy"}:
        return path, raw
    legacy = raw["legacy"]
    if not isinstance(legacy, dict) or set(legacy) != {"task_config"}:
        raise ValueError("legacy config must contain only legacy.task_config")
    task_path = (path.parent / legacy["task_config"]).resolve()
    return task_path, _read_toml(task_path)


def _check_task_layer(path: Path, raw: Mapping[str, Any]) -> None:
    """Reject task files that would shadow the canonical baseline."""
    unknown = set(raw) - set(_TASK_LAYER_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: task configs may only define [task] and [data]; "
            f"move {sorted(unknown)} to baseline.toml or an experiment override"
        )
    for section, allowed in _TASK_LAYER_KEYS.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{path}: [{section}] must be a table")
        extra = set(values) - allowed
        if extra:
            raise ValueError(
                f"{path}: [{section}] keys {sorted(extra)} are not task-specific; "
                "move them to baseline.toml or an experiment override"
            )


def _default_baseline(task_path: Path) -> Path:
    candidates = (task_path.parent / "baseline.toml", task_path.parent.parent / "baseline.toml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find baseline.toml for {task_path}")


def load(
    path: str | Path,
    overrides: Sequence[str] = (),
    *,
    baseline: str | Path | None = None,
    experiment: str | Path | Mapping[str, Any] | None = None,
    experiment_value: Any | None = None,
) -> Config:
    """Resolve baseline -> task -> experiment -> runtime overrides."""
    task_path = Path(path).resolve()
    task_path, task_raw = _resolve_legacy_task(task_path, _read_toml(task_path))
    _check_task_layer(task_path, task_raw)
    baseline_path = Path(baseline).resolve() if baseline is not None else _default_baseline(task_path)
    raw = _deep_merge(_read_toml(baseline_path), task_raw)

    if experiment is None and experiment_value is not None:
        raise ValueError("experiment_value requires an experiment")
    spec: ExperimentSpec | None = None
    if experiment is not None:
        if isinstance(experiment, Mapping):
            experiment_layer = dict(experiment)
        else:
            experiment_path = Path(experiment).resolve()
            experiment_raw = _read_toml(experiment_path)
            if "experiment" in experiment_raw:
                spec = _parse_experiment(experiment_raw)
                if experiment_value is None:
                    raise ValueError(f"experiment {spec.name!r} requires an experiment value")
                experiment_value = match_experiment_value(spec, experiment_value)
                experiment_layer: dict[str, Any] = {}
                for key, value in spec.fixed.items():
                    _set_dotted(experiment_layer, key, copy.deepcopy(value))
                _set_dotted(experiment_layer, spec.variable, experiment_value)
            else:
                if experiment_value is not None:
                    raise ValueError(
                        "experiment_value requires an experiment spec with an [experiment] table"
                    )
                experiment_layer = experiment_raw
        raw = _deep_merge(raw, experiment_layer)

    for item in overrides:
        key, value = _parse_override(item)
        if spec is not None and key == spec.variable:
            # Overriding the declared variable would silently leave the grid:
            # the run would be labelled with one value but train with another.
            raise ValueError(
                f"{key} is the variable of experiment {spec.name!r}; "
                "choose it with the experiment value instead of an override"
            )
        if spec is not None and key in spec.fixed:
            raise ValueError(
                f"{key} is fixed to {spec.fixed[key]!r} by experiment {spec.name!r}; "
                "edit the experiment spec instead of overriding it"
            )
        _set_dotted(raw, key, value)
    return from_dict(raw)


def data_root_override(data_root: str | Path) -> str:
    """Runtime override string for a ``--data-root`` value.

    ``load_run`` and the run planner must resolve ``data.root`` identically,
    so the formatting lives here instead of in either caller. ``json.dumps``
    quotes the path, so a root with spaces or TOML-significant characters
    cannot be mis-parsed by the ordinary override parser.
    """
    return f"data.root={json.dumps(str(data_root))}"


def load_run(
    task: str | Path,
    overrides: Sequence[str] = (),
    *,
    experiment: str | Path | None = None,
    experiment_value: Any | None = None,
    num_demos: int | None = None,
    seed: int | None = None,
    data_root: str | Path | None = None,
) -> Config:
    """Resolve one training run for the entry points.

    ``--num-demos``, ``--seed`` and ``--data-root`` are ordinary runtime
    overrides applied after ``--set``, so they go through the same experiment
    variable check as every other override.
    """
    runtime = list(overrides)
    if num_demos is not None:
        runtime.append(f"data.num_demos={int(num_demos)}")
    if seed is not None:
        runtime.append(f"train.seed={int(seed)}")
    if data_root is not None:
        runtime.append(data_root_override(data_root))
    return load(task, runtime, experiment=experiment, experiment_value=experiment_value)
