"""Silver-vs-gold acceptance gate (tech-spec §6.4).

Implements the narrowest slice of the three-layer benchmark harness
(tech-spec §6): the hard runtime assertion that "the eval harness refuses
to compute an acceptance report against a dataset containing any
non-gold-tagged item in its test split." Does NOT implement real metric
computation (macro-F1, confusion-pair F1, chrF/chrF++/COMET, perplexity —
tech-spec §6.1), leakage/contamination controls (§6.2), or the conflation
red-team suite (§6.3, already a named seam — run_red_team — in
src/bakeoff/). This module's job is the gate every future metric
implementation must pass through, not the metrics themselves — same
seam-not-implementation pattern as src/bakeoff/'s fine_tune/score/
run_red_team callables.

compute_acceptance_report() is the hard-gated path: it raises
AcceptanceGateError, naming every offending item, before ever invoking the
injected compute_metrics callable on a non-gold split — tier: silver
labels (LID auto-labels, back-translation output, any noisy
externally-sourced rows) are usable for training/bootstrapping only, never
final acceptance. compute_calibration_report() is a separate, structurally
distinct entry point for Layer-A transfer-calibrator runs, which the
tech-spec explicitly permits to use silver or mixed-tier data — its
CalibrationReport return type is a different type from AcceptanceReport,
so a calibration result can never be silently passed off as an acceptance
decision downstream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from data_contract import LanguageTag

Layer = Literal["A", "B", "C"]
Tier = Literal["gold", "silver"]


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One item in a test split, declaring its own tier and layer rather
    than leaving the eval code to infer either."""

    item_id: str
    language_tag: LanguageTag
    tier: Tier
    layer: Layer


class AcceptanceGateError(ValueError):
    """Raised when compute_acceptance_report() is given a test split
    containing any non-gold item, or an empty split. Never raised by
    compute_calibration_report(), which has no such gate."""


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    """The result of a gold-only acceptance run — tech-spec §6.4's "real
    acceptance test." Structurally distinct from CalibrationReport."""

    language: LanguageTag
    metrics: dict[str, float]
    item_count: int


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """The result of a Layer-A calibration run, permitted to use silver or
    mixed-tier data. Structurally distinct from AcceptanceReport — never
    accidentally interchangeable with an acceptance decision."""

    language: LanguageTag
    metrics: dict[str, float]
    item_count: int


def compute_acceptance_report(
    language: LanguageTag,
    test_split: list[EvalItem],
    *,
    compute_metrics: Callable[[list[EvalItem], LanguageTag], dict[str, float]],
) -> AcceptanceReport:
    """The hard-gated entry point. Raises AcceptanceGateError — without
    ever invoking `compute_metrics` — if `test_split` is empty or contains
    any item whose tier is not "gold"."""
    if not test_split:
        raise AcceptanceGateError("test_split is empty; refusing to compute an acceptance report")

    non_gold_ids = [item.item_id for item in test_split if item.tier != "gold"]
    if non_gold_ids:
        raise AcceptanceGateError(
            "test_split contains non-gold item(s), refusing to compute an acceptance report "
            f"(tech-spec §6.4): {non_gold_ids}"
        )

    metrics = compute_metrics(test_split, language)
    return AcceptanceReport(language=language, metrics=metrics, item_count=len(test_split))


def compute_calibration_report(
    language: LanguageTag,
    test_split: list[EvalItem],
    *,
    compute_metrics: Callable[[list[EvalItem], LanguageTag], dict[str, float]],
) -> CalibrationReport:
    """Layer-A calibration path: no gold-only gate. tech-spec §6: "reported,
    never final acceptance" — enforced by CalibrationReport's distinct type,
    not by any restriction on which tiers may appear here."""
    metrics = compute_metrics(test_split, language)
    return CalibrationReport(language=language, metrics=metrics, item_count=len(test_split))
