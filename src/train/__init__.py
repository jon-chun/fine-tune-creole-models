"""LoRA/QLoRA/DoRA hyperparameter config loader (tech-spec §5).

tech-spec §5's hyperparameter table is fully decided and framed as "starting
points to validate empirically per language via the bake-off (§3.2), not
fixed configuration." This module loads configs/training/lora_defaults.yml
into a typed Hyperparameters object and validates it — it does NOT build the
training driver itself (no Unsloth/PEFT/Axolotl integration, no actual
LoRA/QLoRA/DoRA fine-tuning, no GPU dependency). Hyperparameters is the type
a future training-driver ticket will consume and the type src/bakeoff/'s
fine_tune seam will eventually be wired to produce/accept.

quantization_train is a hard gate: 4-bit is rejected outright, never loaded,
because tech-spec §5 flags it as less stable during adapter merging — a
correctness risk, not a tunable. rank and epochs are soft gates: values
outside the tech-spec's documented ranges load successfully but are recorded
in LoadedTrainingConfig.warnings, since the tech-spec frames both as sweepable
starting points, not hard rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

_VALID_QUANTIZATION = frozenset({"bf16", "fp16", "8bit"})
_RANK_RANGE = range(16, 33)
_EPOCH_RANGE = range(1, 3)

QuantizationMode = Literal["bf16", "fp16", "8bit"]
TargetModules = Literal["all_linear", "attention_qv_only"]


class TrainingConfigError(ValueError):
    """Raised when lora_defaults.yml is malformed or fails a hard-gate check
    (currently: quantization_train outside {bf16, fp16, 8bit})."""


@dataclass(frozen=True, slots=True)
class Hyperparameters:
    """One fully-resolved hyperparameter set. alpha is always 2 * rank
    (tech-spec §5's scaling convention) — never read directly from YAML, so
    it cannot drift out of sync with rank."""

    rank: int
    alpha: int
    target_modules: TargetModules
    learning_rate: float
    epochs: int
    batch_size_strategy: str
    quantization_train: QuantizationMode
    warmup_fraction: float


@dataclass(frozen=True, slots=True)
class LoadedTrainingConfig:
    """A loaded Hyperparameters plus any soft-gate warnings (rank/epochs
    outside the tech-spec §5 documented range). Empty warnings means every
    field was within the documented range."""

    hyperparameters: Hyperparameters
    warnings: list[str]


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
    raise TrainingConfigError(
        f"{path}: quantization_train={raw!r} is not supported — "
        "4-bit is flagged as less stable during adapter merging (tech-spec §5); "
        f"use one of {sorted(_VALID_QUANTIZATION)}"
    )


def load_hyperparameters(path: Path) -> LoadedTrainingConfig:
    """Parse and validate lora_defaults.yml. Raises TrainingConfigError if
    quantization_train is not one of {bf16, fp16, 8bit} (including any
    4-bit/4bit spelling). rank/epochs outside their documented ranges do not
    raise — they're recorded in the returned warnings list instead."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw, dict):
        raise TrainingConfigError(f"{path}: top level must be a mapping")

    quantization = _parse_quantization(raw.get("quantization_train"), path)

    rank = int(raw["rank"])
    epochs = int(raw["epochs"])

    warnings: list[str] = []
    if rank not in _RANK_RANGE:
        warnings.append(
            f"rank={rank} is outside tech-spec §5's documented 16-32 range — "
            "starting point to validate empirically, not a hard rule"
        )
    if epochs not in _EPOCH_RANGE:
        warnings.append(
            f"epochs={epochs} is outside tech-spec §5's documented 1-2 range "
            "(a ceiling to avoid memorization) — starting point to validate empirically"
        )

    hyperparameters = Hyperparameters(
        rank=rank,
        alpha=2 * rank,
        target_modules=raw["target_modules"],
        learning_rate=float(raw["learning_rate"]),
        epochs=epochs,
        batch_size_strategy=raw["batch_size_strategy"],
        quantization_train=quantization,
        warmup_fraction=float(raw["warmup_fraction"]),
    )
    return LoadedTrainingConfig(hyperparameters=hyperparameters, warnings=warnings)


def sweep_learning_rate(
    base_rate: float,
    *,
    orders_of_magnitude: float = 1.0,
    steps: int = 5,
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
