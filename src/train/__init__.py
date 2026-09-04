"""LoRA/QLoRA/DoRA hyperparameter config loader (tech-spec v2 §5; MIG-01d,
issue #28).

tech-spec v2 §5's hyperparameter table frames every value as a "starting
point to validate empirically per language via the bake-off (§3.2), not
fixed configuration." This module loads configs/training/lora_defaults.yml
into a typed Hyperparameters object and validates it — it does NOT build the
training driver itself (no Unsloth/PEFT/Axolotl integration, no actual
LoRA/QLoRA/DoRA fine-tuning, no GPU dependency; backlog 0009). Hyperparameters
is the type a future training-driver ticket will consume and the type
src/bakeoff/'s fine_tune seam will eventually be wired to produce/accept.

quantization_train is a hard gate for every value except `nf4`: bf16/fp16/8bit
are always accepted, and any other spelling (4bit, garbage, ...) is rejected
outright. `nf4` is accepted by the loader itself (tech-spec v2 §5: "4-bit NF4
QLoRA only for ≥20B stretch arms") but the ≥20B-stretch-arm rule is enforced
by `assert_nf4_requires_stretch_arm`, not here — this loader has no
`bakeoff.Candidate` in scope (no cross-sibling import; `src/bakeoff/`'s own
docstring establishes the same rule in the other direction) and so cannot
know the base model's size on its own. rank, epochs, effective_batch_sequences,
replay_fraction, learning_rate and max_seq_len are soft gates: values outside
their tech-spec-documented ranges load successfully but are recorded in
LoadedTrainingConfig.warnings, since the tech-spec frames each as a sweepable
starting point, not a hard rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, SupportsFloat, SupportsInt, cast, get_args

import yaml

from data_contract import DataContractError, LanguageTag, validate_literal

# _VALID_QUANTIZATION: the full closed set the loader accepts, including
# `nf4` (tech-spec v2 §5's ≥20B-stretch-arm addition; MIG-01d). Any other
# spelling — including every 4-bit spelling other than the canonical `nf4` —
# is rejected by `_parse_quantization`.
_VALID_QUANTIZATION = frozenset({"bf16", "fp16", "8bit", "nf4"})

# _RANK_RANGE / _EPOCH_RANGE: v2 widened ranges (tech-spec v2 §5's table;
# MIG-01d, issue #28). Python `range` is exclusive of its stop, so these
# encode the inclusive 8-64 and 1-3 bands, matching the pre-existing
# convention this module already used for the v1 16-32/1-2 ranges.
_RANK_RANGE = range(8, 65)
_EPOCH_RANGE = range(1, 4)

# _EFFECTIVE_BATCH_SEQUENCES_RANGE / _REPLAY_FRACTION_BAND /
# _LEARNING_RATE_BAND / _MAX_SEQ_LEN_RANGE: tech-spec v2 §5's table (MIG-01d).
# Each is a soft gate — a value outside the band loads successfully with a
# warning, mirroring rank/epochs' existing pattern.
_EFFECTIVE_BATCH_SEQUENCES_RANGE = range(16, 33)
_REPLAY_FRACTION_BAND = (0.10, 0.20)
_LEARNING_RATE_BAND = (1e-4, 2e-4)
_MAX_SEQ_LEN_RANGE = (1024, 2048)

# min_base_size_b_for_nf4: the single source for the ≥20B-stretch-arm
# threshold `assert_nf4_requires_stretch_arm` checks (tech-spec v2 §5;
# decision brief R4 Q8 ratification). Named module constant so the number
# lives in exactly one place.
min_base_size_b_for_nf4 = 20

# Default sweep_learning_rate() step count (tech-spec §5: "sweep roughly ±1
# order of magnitude"; the step count itself is an invented parameter with no
# prior YAML home — see lora_defaults.yml's lr_sweep_steps key). Kept as the
# function's own default so sweep_learning_rate() stays pure/path-free;
# load_hyperparameters() reads the YAML key and callers pass it through
# explicitly when they want the configured value instead of this default.
_DEFAULT_SWEEP_STEPS = 5

QuantizationMode = Literal["bf16", "fp16", "8bit", "nf4"]
TargetModules = Literal["all_linear", "attention_qv_only"]

# Method: the training *technique* axis (tech-spec v2 §5: "Method | LoRA on
# all linear layers [DoRA; attention-only negative control]"), distinct from
# `target_modules` which continues to name *which* layers `method="lora"`
# targets (MIG-01d, issue #28). Judgment call: `method="attention_only"` does
# NOT automatically imply `target_modules="attention_qv_only"` — both fields
# are validated independently and a caller must set both consistently. This
# keeps the loader a pure structural validator (no cross-field inference)
# and matches how rank/alpha's own derivation is the only place this module
# currently computes one field from another; see
# test_method_attention_only_does_not_imply_target_modules.
Method = Literal["lora", "dora", "attention_only"]

_TARGET_MODULES = frozenset(get_args(TargetModules))
_METHODS = frozenset(get_args(Method))


class TrainingConfigError(ValueError):
    """Raised when lora_defaults.yml is malformed, is missing a required
    key, or fails a hard-gate check (quantization_train outside {bf16, fp16,
    8bit, nf4}; target_modules or method outside its closed set;
    batch_size_strategy empty; seeds empty; early_stopping.patience < 1)
    (issue #15; issue #28)."""


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    """Dev-split early-stopping config (tech-spec v2 §5: "early stopping on
    a dialect dev split (patience 1 epoch on dev chrF++ / probe accuracy),
    never on the locked gold"). `metric` names the dev-split metric to watch
    (e.g. "dev_chrf_plus_plus" or "probe_accuracy") — this loader carries the
    string through, it does not interpret it. `patience` is in epochs and
    must be >= 1 when present (issue #28)."""

    metric: str
    patience: int


@dataclass(frozen=True, slots=True)
class Hyperparameters:
    """One fully-resolved hyperparameter set. alpha is always 2 * rank
    (tech-spec §5's scaling convention) — never read directly from YAML, so
    it cannot drift out of sync with rank.

    max_seq_len, batch_token_budget, and max_train_tokens are corpus-derived
    cost/memory controls (tech-spec §10 "Cost and memory controls"; issue
    #20). None of the three has an invented repo-wide default: each must
    come from `overrides.<language>` in lora_defaults.yml, populated by
    running utils/fine-tune-cajun-corpus-stats.py against the real corpus
    (max_seq_len, max_train_tokens) plus a hardware probe (batch_token_budget
    — memory-bound, found empirically on the real GPU; backlog 0009 defers
    that probe to the training driver itself). `None` is the honest
    not-yet-derived state when no `language` is requested; requesting a
    `language` without all three present raises TrainingConfigError.

    method / effective_batch_sequences / replay_fraction / seeds /
    early_stopping / task_prefixes are v2 additions (tech-spec v2 §5;
    MIG-01d, issue #28). `effective_batch_sequences` replaces v1's free-string
    `batch_size_strategy` outright (no deprecated alias — same "hard rename,
    no compatibility alias" posture MIG-01a takes for `Tier`→`data_class`,
    since this is an internal type with no consumer outside this repo's own
    tests; decision brief §3 item 6 lists it as wholesale-superseded, not
    extended)."""

    rank: int
    alpha: int
    method: Method
    target_modules: TargetModules
    learning_rate: float
    epochs: int
    effective_batch_sequences: int
    quantization_train: QuantizationMode
    warmup_fraction: float
    seeds: tuple[int, ...]
    replay_fraction: float
    early_stopping: EarlyStoppingConfig | None
    task_prefixes: tuple[str, ...]
    max_seq_len: int | None
    batch_token_budget: int | None
    max_train_tokens: int | None


@dataclass(frozen=True, slots=True)
class LoadedTrainingConfig:
    """A loaded Hyperparameters plus any soft-gate warnings (rank, epochs,
    effective_batch_sequences, replay_fraction, learning_rate, max_seq_len
    outside their tech-spec v2 §5 documented ranges). Empty warnings means
    every field was within its documented range."""

    hyperparameters: Hyperparameters
    warnings: list[str]
    lr_sweep_steps: int


def _normalize_quantization(raw: object) -> str:
    return str(raw).strip().lower().replace("-", "")


def _parse_quantization(raw: object, path: Path) -> QuantizationMode:
    normalized = _normalize_quantization(raw)
    if normalized == "bf16":
        return "bf16"
    if normalized == "fp16":
        return "fp16"
    if normalized == "8bit":
        return "8bit"
    if normalized == "nf4":
        return "nf4"
    raise TrainingConfigError(
        f"{path}: quantization_train={raw!r} is not supported — "
        f"use one of {sorted(_VALID_QUANTIZATION)} "
        "(nf4 is accepted here but only usable for ≥20B stretch arms — see "
        "assert_nf4_requires_stretch_arm, tech-spec v2 §5)"
    )


def assert_nf4_requires_stretch_arm(
    quantization: QuantizationMode, *, base_size_b: int | None
) -> None:
    """Enforce the ≥20B-stretch-arm rule for `nf4` quantization (tech-spec v2
    §5: "4-bit NF4 QLoRA only for ≥20B stretch arms, adapters unmerged or
    merged onto BF16 weights"; decision brief R4 Q8 ratification; MIG-01d,
    issue #28).

    A separate exported function rather than a `load_hyperparameters` check:
    the loader alone never has a `bakeoff.Candidate` (and therefore never a
    base model size) in scope, and `src/train/` must not import
    `src/bakeoff/`'s `Candidate` type directly (no cross-sibling import — the
    same ground rule `src/bakeoff/`'s own docstring cites for not importing
    `train.Hyperparameters`). The future training-driver ticket (backlog
    0009) calls this at the point where both a loaded `Hyperparameters` and a
    `Candidate` are in scope simultaneously, passing `candidate.size_b` as
    `base_size_b`.

    Raises TrainingConfigError when `quantization == "nf4"` and
    `base_size_b` is `None` or below `min_base_size_b_for_nf4`. Every other
    quantization mode is a no-op regardless of `base_size_b` — the ≥20B rule
    only constrains `nf4`."""
    if quantization != "nf4":
        return
    if base_size_b is None:
        raise TrainingConfigError(
            "quantization_train='nf4' requires a known base model size to check "
            f"against the ≥{min_base_size_b_for_nf4}B stretch-arm rule (tech-spec "
            "v2 §5), but base_size_b is None"
        )
    if base_size_b < min_base_size_b_for_nf4:
        raise TrainingConfigError(
            f"quantization_train='nf4' requires base_size_b >= "
            f"{min_base_size_b_for_nf4} (tech-spec v2 §5's ≥20B stretch-arm rule), "
            f"got base_size_b={base_size_b}"
        )


def _parse_optional_positive_int(merged: dict[str, object], key: str, path: Path) -> int | None:
    """Read an optional corpus-derived knob (max_seq_len, batch_token_budget,
    max_train_tokens). Absent or explicit `null` -> None (not yet derived).
    Present and <= 0 raises — a non-positive token budget/length/ceiling is
    never a valid value, invented default or not."""
    raw = merged.get(key)
    if raw is None:
        return None
    value = int(cast(SupportsInt, raw))
    if value <= 0:
        raise TrainingConfigError(f"{path}: {key}={value} must be a positive integer")
    return value


def _parse_seeds(raw: object, path: Path) -> tuple[int, ...]:
    """Parse the required `seeds` list (tech-spec v2 §5: "3 for primary arms
    ... single-seed = exploratory"; MIG-01d, issue #28). Any non-empty list
    of ints is accepted structurally — 3 is the documented primary-arm count,
    not a hard-enforced length, since a single-seed exploratory run is named
    explicitly as a valid (if non-primary) use. An empty list is never valid:
    a configured seed list with nothing in it cannot back any run at all."""
    if not isinstance(raw, list) or not raw:
        raise TrainingConfigError(
            f"{path}: seeds must be a non-empty list of ints, got {raw!r}"
        )
    return tuple(int(cast(SupportsInt, item)) for item in raw)


def _parse_task_prefixes(raw: object, path: Path) -> tuple[str, ...]:
    """Parse `task_prefixes` (tech-spec v2 §5's single-stage SFT regime list;
    MIG-01d, issue #28). The loader carries these strings through without
    interpreting them (a training-driver concern) and validates only that
    the list is non-empty when present — no closed enum, since this list is
    expected to grow per-language."""
    if not isinstance(raw, list) or not raw:
        raise TrainingConfigError(
            f"{path}: task_prefixes must be a non-empty list of strings, got {raw!r}"
        )
    return tuple(str(item) for item in raw)


def _parse_early_stopping(raw: object, path: Path) -> EarlyStoppingConfig | None:
    """Parse the optional `early_stopping` block (tech-spec v2 §5: "patience
    1 epoch on dev chrF++ / probe accuracy... never on the locked gold";
    MIG-01d, issue #28). Absent -> None (no early stopping configured).
    `patience` must be >= 1 when present."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TrainingConfigError(f"{path}: early_stopping must be a mapping, got {raw!r}")
    try:
        metric = str(raw["metric"])
        patience = int(cast(SupportsInt, raw["patience"]))
    except KeyError as exc:
        raise TrainingConfigError(f"{path}: early_stopping missing required field {exc}") from exc
    if patience < 1:
        raise TrainingConfigError(
            f"{path}: early_stopping.patience={patience} must be >= 1 epoch"
        )
    return EarlyStoppingConfig(metric=metric, patience=patience)


def load_hyperparameters(path: Path, *, language: LanguageTag | None = None) -> LoadedTrainingConfig:
    """Parse and validate lora_defaults.yml. Raises TrainingConfigError if
    quantization_train is not one of {bf16, fp16, 8bit, nf4} (including any
    other 4-bit spelling), if a required field (seeds, method,
    effective_batch_sequences, replay_fraction, task_prefixes, ...) is
    missing or malformed, or if early_stopping.patience < 1 when present.
    rank/epochs/effective_batch_sequences/replay_fraction/learning_rate/
    max_seq_len outside their documented ranges do not raise — they're
    recorded in the returned warnings list instead.

    When `language` is given and the YAML has an `overrides: {<language>:
    {...}}` block, those keys are applied on top of the top-level fields
    before validation — tech-spec §5's "validated per language" (§7 gap:
    lora_defaults.yml was previously one file for both languages).

    max_seq_len/batch_token_budget/max_train_tokens (issue #20) load as
    `None` when absent from both the top level and the applied override —
    the honest not-yet-derived state. When `language` is given, all three
    must resolve to a value after overrides are applied, or this raises
    TrainingConfigError naming the language and the missing key(s). This
    corpus-derived-knob requirement is unchanged by MIG-01d (issue #28 Out of
    scope item 11)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw, dict):
        raise TrainingConfigError(f"{path}: top level must be a mapping")

    merged: dict[str, object] = dict(raw)
    if language is not None:
        overrides = raw.get("overrides")
        if isinstance(overrides, dict):
            language_overrides = overrides.get(language)
            if isinstance(language_overrides, dict):
                merged.update(language_overrides)

    quantization = _parse_quantization(merged.get("quantization_train"), path)

    seeds = _parse_seeds(merged.get("seeds"), path)

    try:
        rank = int(cast(SupportsInt, merged["rank"]))
        epochs = int(cast(SupportsInt, merged["epochs"]))
        method_raw = merged["method"]
        target_modules_raw = merged["target_modules"]
        learning_rate = float(cast(SupportsFloat, merged["learning_rate"]))
        effective_batch_sequences = int(cast(SupportsInt, merged["effective_batch_sequences"]))
        warmup_fraction = float(cast(SupportsFloat, merged["warmup_fraction"]))
        replay_fraction = float(cast(SupportsFloat, merged["replay_fraction"]))
    except KeyError as exc:
        raise TrainingConfigError(f"{path}: missing required field {exc}") from exc

    try:
        method = validate_literal(method_raw, tuple(_METHODS), "method")
    except DataContractError as exc:
        raise TrainingConfigError(f"{path}: {exc}") from exc

    try:
        target_modules = validate_literal(
            target_modules_raw, tuple(_TARGET_MODULES), "target_modules"
        )
    except DataContractError as exc:
        raise TrainingConfigError(f"{path}: {exc}") from exc

    task_prefixes = _parse_task_prefixes(merged.get("task_prefixes"), path)
    early_stopping = _parse_early_stopping(merged.get("early_stopping"), path)

    max_seq_len = _parse_optional_positive_int(merged, "max_seq_len", path)
    batch_token_budget = _parse_optional_positive_int(merged, "batch_token_budget", path)
    max_train_tokens = _parse_optional_positive_int(merged, "max_train_tokens", path)

    if language is not None:
        missing_knobs = [
            name
            for name, value in (
                ("max_seq_len", max_seq_len),
                ("batch_token_budget", batch_token_budget),
                ("max_train_tokens", max_train_tokens),
            )
            if value is None
        ]
        if missing_knobs:
            raise TrainingConfigError(
                f"{path}: language={language!r} is missing corpus-derived knob(s) "
                f"{missing_knobs} — run utils/fine-tune-cajun-corpus-stats.py against "
                f"the {language!r} corpus and fill overrides.{language} in {path.name} "
                "(tech-spec §10 Cost and memory controls)"
            )

    warnings: list[str] = []
    if rank not in _RANK_RANGE:
        warnings.append(
            f"rank={rank} is outside tech-spec v2 §5's documented 8-64 range — "
            "starting point to validate empirically, not a hard rule"
        )
    if epochs not in _EPOCH_RANGE:
        warnings.append(
            f"epochs={epochs} is outside tech-spec v2 §5's documented 1-3 range "
            "(a ceiling to avoid memorization) — starting point to validate empirically"
        )
    if effective_batch_sequences not in _EFFECTIVE_BATCH_SEQUENCES_RANGE:
        warnings.append(
            f"effective_batch_sequences={effective_batch_sequences} is outside tech-spec "
            "v2 §5's documented 16-32 range — starting point to validate empirically"
        )
    if not (_REPLAY_FRACTION_BAND[0] <= replay_fraction <= _REPLAY_FRACTION_BAND[1]):
        warnings.append(
            f"replay_fraction={replay_fraction} is outside tech-spec v2 §5's documented "
            f"{_REPLAY_FRACTION_BAND[0]}-{_REPLAY_FRACTION_BAND[1]} band — ablation range is "
            "0.0-0.30, so this is still a valid exploratory value"
        )
    if not (_LEARNING_RATE_BAND[0] <= learning_rate <= _LEARNING_RATE_BAND[1]):
        warnings.append(
            f"learning_rate={learning_rate} is outside tech-spec v2 §5's documented "
            f"{_LEARNING_RATE_BAND[0]}-{_LEARNING_RATE_BAND[1]} range — ±1 order sweep is "
            "an explicit ablation, not a hard rule"
        )
    if max_seq_len is not None and not (
        _MAX_SEQ_LEN_RANGE[0] <= max_seq_len <= _MAX_SEQ_LEN_RANGE[1]
    ):
        warnings.append(
            f"max_seq_len={max_seq_len} is outside tech-spec v2 §5's documented "
            f"{_MAX_SEQ_LEN_RANGE[0]}-{_MAX_SEQ_LEN_RANGE[1]} token range (annex §7) — "
            "corpus-derived value to validate, not a hard rule"
        )
    if (
        max_train_tokens is not None
        and batch_token_budget is not None
        and max_train_tokens < batch_token_budget
    ):
        warnings.append(
            f"max_train_tokens={max_train_tokens} is less than batch_token_budget="
            f"{batch_token_budget} — the run would stop before a single full step's "
            "worth of tokens is consumed"
        )

    hyperparameters = Hyperparameters(
        rank=rank,
        alpha=2 * rank,
        method=cast(Method, method),
        target_modules=cast(TargetModules, target_modules),
        learning_rate=learning_rate,
        epochs=epochs,
        effective_batch_sequences=effective_batch_sequences,
        quantization_train=quantization,
        warmup_fraction=warmup_fraction,
        seeds=seeds,
        replay_fraction=replay_fraction,
        early_stopping=early_stopping,
        task_prefixes=task_prefixes,
        max_seq_len=max_seq_len,
        batch_token_budget=batch_token_budget,
        max_train_tokens=max_train_tokens,
    )
    lr_sweep_steps = int(cast(SupportsInt, merged.get("lr_sweep_steps", _DEFAULT_SWEEP_STEPS)))
    return LoadedTrainingConfig(
        hyperparameters=hyperparameters, warnings=warnings, lr_sweep_steps=lr_sweep_steps
    )


def sweep_learning_rate(
    base_rate: float,
    *,
    orders_of_magnitude: float = 1.0,
    steps: int = _DEFAULT_SWEEP_STEPS,
) -> list[float]:
    """Log-spaced learning-rate sweep spanning ±orders_of_magnitude around
    base_rate (tech-spec §5: "sweep roughly ±1 order of magnitude"). Always
    includes base_rate itself. Pure and deterministic."""
    if steps < 2:
        raise ValueError("steps must be >= 2")

    log_base = math.log10(base_rate)
    log_low = log_base - orders_of_magnitude
    log_high = log_base + orders_of_magnitude
    step_size = (log_high - log_low) / (steps - 1)

    values = [10 ** (log_low + i * step_size) for i in range(steps)]
    if base_rate not in values:
        values.append(base_rate)
        values.sort()
    return values
