"""Silver-vs-gold acceptance gate (tech-spec v2 §6.4).

Implements a narrow slice of the three-layer benchmark harness (tech-spec v2
§6): the hard runtime assertion that "the eval harness refuses to compute an
acceptance report against a dataset containing any non-gold item in its test
split", now under the `data_class` name (gold/silver/bronze/synthetic —
MIG-01e, issue #29), plus three tech-spec v2 additions this module is
well-placed to host: the recorded `ModelReleaseClass` on an acceptance report
(§7), the forgetting-axis flag (§6.1), and the Class 0 invariant hooks
(§6.3). Does NOT implement real metric computation (macro-F1, confusion-pair
F1, chrF/chrF++/COMET, perplexity — tech-spec v2 §6.1), leakage/contamination
controls (§6.2), or the conflation red-team suite itself (§6.3, a named seam
— run_red_team — in src/bakeoff/, MIG-01c). This module's job is the gate
every future metric implementation must pass through, not the metrics
themselves — same seam-not-implementation pattern as src/bakeoff/'s
fine_tune/score/run_red_team callables.

compute_acceptance_report() is the hard-gated path: it raises
AcceptanceGateError, naming every offending item, before ever invoking the
injected compute_metrics callable on a disqualified split — data_class:
silver/bronze labels (LID auto-labels, back-translation output, unreviewed
harvested text, any noisy externally-sourced rows) are usable for
training/bootstrapping only, never final acceptance; data_class: synthetic
never enters an evaluation assembly at all (tech-spec v2 §6.4).
compute_calibration_report() is a separate, structurally distinct entry
point for Layer-A transfer-calibrator runs, which the tech-spec explicitly
permits to use silver or mixed-class data — its CalibrationReport return
type is a different type from AcceptanceReport, so a calibration result can
never be silently passed off as an acceptance decision downstream.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

from data_contract import DataClass, LanguageTag, ModelReleaseClass, validate_literal
from eval.metrics import MetricReport

Layer = Literal["A", "B", "C"]

# DataClass is data_contract's own gold/silver/bronze/synthetic partition
# (MIG-01a, issue #25; originally the two-value-plus-synthetic `Tier` from
# issue #11). Re-exported here rather than redefined, so src.eval never
# disagrees with data_contract's single owner of the name — a hard rename
# from the v1 `Tier`/`tier`, no compatibility alias (issue #29 decision:
# `EvalItem(tier=...)` is a TypeError, not a deprecated kwarg).
__all__ = [
    "AcceptanceGateError",
    "AcceptanceReport",
    "CalibrationReport",
    "DataClass",
    "EvalItem",
    "Layer",
    "MetricReport",
    "check_class_0_invariants",
    "compute_acceptance_report",
    "compute_calibration_report",
    "flag_forgetting_regression",
]

_LAYERS = frozenset(get_args(Layer))
_DATA_CLASSES = frozenset(get_args(DataClass))


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One item in a test split, declaring its own data_class and layer
    rather than leaving the eval code to infer either.

    `synthetic` (issue #29 story 7): added for defense-in-depth against a
    synthetic item silently entering a gold-tagged acceptance split. This
    mirrors `DataItem.synthetic`'s own agreement invariant with its
    `data_class`, but `EvalItem` is a separate, smaller type with no
    connection to `DataItem` at construction time — this module cannot rely
    solely on upstream `DataItem`-level enforcement (tech-spec v2 §6.3 Class
    0 table; `check_class_0_invariants` below is the check this field
    feeds).

    `source_text`/`reference`/`reference_diplomatic` (issue #34, tech-spec
    v2 §6.1; issue #23 row D2 "EvalItem has no text/reference"): text an
    item carries for scoring. `source_text` is the input side of a
    translation/normalization pair (`None` for a task with no distinct
    source, e.g. a contrast/judge probe scored against its own expected
    output). `reference` is the normalized-layer gold text — the layer
    every headline metric in `eval.metrics` scores against by default.
    `reference_diplomatic` is the separate verbatim/diplomatic-layer
    reference tech-spec v2 §9's cWER/sWER decomposition needs
    (`eval.metrics.wer_cer_decomposed`); `None` when an item has no
    diplomatic layer distinct from its normalized one. `cell_id` (tech-spec
    v2 §4's diff-catalog cell id, e.g. "VERB-001"; tech-spec v2 §6.3's
    probe-schema field of the same name) is the grouping key
    `eval.metrics.per_cell_accuracy_with_wilson_ci` needs for
    contrast/judge tasks; `None` for tasks with no per-cell breakdown (MT,
    normalize). All four fields default to `None` so every existing
    `EvalItem(...)` call site in this module's own tests keeps working
    unchanged (issue #34 acceptance: "all 34 existing eval tests still
    pass").
    """

    item_id: str
    language_tag: LanguageTag
    data_class: DataClass
    layer: Layer
    synthetic: bool = False
    source_text: str | None = None
    reference: str | None = None
    reference_diplomatic: str | None = None
    cell_id: str | None = None

    def __post_init__(self) -> None:
        # Same enum-membership discipline as DataItem's own __post_init__
        # (issue #11 proposed resolution #4, carried forward under the v2
        # name): a bad data_class/layer value must be rejected at
        # construction, not silently accepted and only noticed later by the
        # acceptance gate.
        validate_literal(self.data_class, tuple(_DATA_CLASSES), "data_class")
        validate_literal(self.layer, tuple(_LAYERS), "layer")


class AcceptanceGateError(ValueError):
    """Raised when compute_acceptance_report() is given a test split
    containing any non-gold item, any item outside Layer B, an empty split,
    or a Class 0 invariant violation (tech-spec v2 §6.3). Never raised by
    compute_calibration_report(), which has no such gate."""


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    """The result of a gold-only, Layer-B acceptance run — tech-spec v2
    §6.4's "real acceptance test." Structurally distinct from
    CalibrationReport. `layer` is always "B": every report declares which
    layer it belongs to (CONTEXT.md "Three-layer evaluation"), rather than
    leaving the reader to infer it from the report type alone.

    `release_class` (tech-spec v2 §7, ModelReleaseClass; issue #29 story 6):
    a plain recorded field, not derived here. `src/eval` does not run the
    red-team suite (`src/bakeoff/`'s job, MIG-01c's
    `derive_model_release_class(verdict, *, license_lineage_clear)` from a
    `RedTeamVerdict`) — this module is a pure recorder of a release-class
    decision made upstream, so the bake-off's derivation and this report can
    never disagree about which release class applies.

    `forgetting_axis_flagged`/`forgetting_axis_delta` (tech-spec v2 §6.1,
    "Forgetting axis, every arm"): populated from `flag_forgetting_regression`
    when the caller supplies forgetting-axis metrics to
    `compute_acceptance_report`; `forgetting_axis_delta` is `None` when no
    forgetting-axis metrics were supplied for this run (an acceptance run is
    not required to include a forgetting-axis comparison every time it is
    called, e.g. a first baseline run with no prior arm to compare against).
    """

    language: LanguageTag
    metrics: dict[str, float]
    item_count: int
    release_class: ModelReleaseClass
    layer: Layer = "B"
    forgetting_axis_flagged: bool = False
    forgetting_axis_delta: float | None = None


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """The result of a Layer-A calibration run, permitted to use silver,
    bronze or mixed-class data. Structurally distinct from AcceptanceReport
    — never accidentally interchangeable with an acceptance decision.
    `layer` defaults to "A" (tech-spec v2 §6's transfer-calibrator layer) but
    is a real field, not a hardcoded assumption, since a caller could in
    principle reuse this path for a differently-layered calibration run.

    No `release_class` or forgetting-axis fields (issue #29 story 9,
    explicit-not-implicit decision): a Layer-A calibrator is a published
    external baseline, not a model this project releases, so
    `ModelReleaseClass` does not apply to it; and Layer-A runs have no
    project-internal "before" arm of their own to compare a forgetting delta
    against (the comparison tech-spec v2 §6.1 names is always against a
    prior arm of the same trained candidate, which only Layer-B/C runs
    produce). Both omissions are recorded here rather than left for a later
    reader to wonder whether they were forgotten.
    """

    language: LanguageTag
    metrics: dict[str, float]
    item_count: int
    layer: Layer = "A"


def flag_forgetting_regression(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    bootstrap_half_width: dict[str, float],
) -> bool:
    """Pure threshold rule for the forgetting axis (tech-spec v2 §6.1): "flag
    if any metric drops > 2 points absolute or > the bootstrap half-width,
    whichever is larger."

    `baseline`/`candidate` are Standard-French matched-item metrics keyed by
    metric name (e.g. "chrf", "macro_f1"); `bootstrap_half_width` gives each
    metric's half-width, keyed the same way. A metric present in `candidate`
    but absent from `baseline` (or vice versa) contributes no flag by
    itself — this function only compares metrics both dicts share, since a
    metric neither run measured cannot regress. A metric missing from
    `bootstrap_half_width` falls back to a half-width of 0.0 (the 2-point
    absolute floor alone still applies), rather than raising, so a caller
    that has not yet computed a bootstrap CI for a given metric still gets a
    meaningful (if less precise) answer instead of a crash.

    Kept as a standalone pure function, not inlined into
    `compute_acceptance_report`, so the exact threshold rule is
    independently unit-testable (issue #29 acceptance criteria).
    """
    for metric_name, baseline_value in baseline.items():
        if metric_name not in candidate:
            continue
        candidate_value = candidate[metric_name]
        drop = baseline_value - candidate_value
        half_width = bootstrap_half_width.get(metric_name, 0.0)
        threshold = max(2.0, half_width)
        if drop > threshold:
            return True
    return False


def check_class_0_invariants(
    test_split: list[EvalItem],
    *,
    outputs: list[str] | None = None,
    requested_languages: list[LanguageTag] | None = None,
    output_languages: list[LanguageTag] | None = None,
) -> list[str]:
    """Check the four deterministic pipeline-property invariants named in
    tech-spec v2 §6.3's Class 0 table, returning every violation found (an
    empty list means no Class 0 violation). No I/O, no hidden model calls
    (issue #29 story 10): the NFC and language-match checks accept
    pre-computed inputs (`outputs`, `requested_languages`,
    `output_languages`) rather than running normalization or LID
    themselves, consistent with this module's existing seam-not-
    implementation pattern for `compute_metrics`.

    The four invariants (tech-spec v2 §6.3):
    1. Non-gold item in an acceptance split — the same "not gold, not Layer
       B" predicate `compute_acceptance_report` already gates on, exposed
       here as a reusable, independently callable check.
    2. Synthetic item in a gold-tagged split — `data_class == "gold" and
       synthetic`, using `EvalItem.synthetic` (issue #29 story 7).
    3. Diacritic-stripping or NFC decomposition on a copy/normalize probe —
       checked via `unicodedata.normalize("NFC", s) == s` per output string,
       only when `outputs` is supplied (not every acceptance run has
       copy/normalize probes in scope).
    4. Output language != requested language on a fixed-language task —
       checked by comparing `requested_languages` against
       `output_languages` positionally, only when both are supplied.

    `outputs`/`requested_languages`/`output_languages` are independently
    optional: a caller with no copy/normalize probes in this run simply
    omits `outputs`, and a caller with no fixed-language task in this run
    omits the language pair, without affecting the other checks or the
    non-gold/synthetic-in-gold checks (which always run against
    `test_split`).
    """
    violations: list[str] = []

    disqualified_ids = [
        item.item_id for item in test_split if item.data_class != "gold" or item.layer != "B"
    ]
    if disqualified_ids:
        violations.append(
            "non-gold and/or non-Layer-B item(s) in acceptance split: "
            f"{disqualified_ids}"
        )

    synthetic_in_gold_ids = [
        item.item_id for item in test_split if item.data_class == "gold" and item.synthetic
    ]
    if synthetic_in_gold_ids:
        violations.append(f"synthetic item(s) tagged gold data_class: {synthetic_in_gold_ids}")

    if outputs is not None:
        non_nfc = [text for text in outputs if unicodedata.normalize("NFC", text) != text]
        if non_nfc:
            violations.append(
                f"diacritic-stripping or NFC decomposition detected on {len(non_nfc)} output(s)"
            )

    if requested_languages is not None and output_languages is not None:
        mismatches = [
            (requested, actual)
            for requested, actual in zip(requested_languages, output_languages, strict=True)
            if requested != actual
        ]
        if mismatches:
            violations.append(
                f"output language != requested language on fixed-language task: {mismatches}"
            )

    return violations


def compute_acceptance_report(
    language: LanguageTag,
    test_split: list[EvalItem],
    *,
    compute_metrics: Callable[[list[EvalItem], LanguageTag], dict[str, float] | MetricReport],
    release_class: ModelReleaseClass,
    outputs: list[str] | None = None,
    requested_languages: list[LanguageTag] | None = None,
    output_languages: list[LanguageTag] | None = None,
    forgetting_axis_metrics: dict[str, float] | None = None,
    forgetting_axis_baseline: dict[str, float] | None = None,
    forgetting_axis_bootstrap_half_width: dict[str, float] | None = None,
) -> AcceptanceReport:
    """The hard-gated entry point. Raises AcceptanceGateError — without ever
    invoking `compute_metrics` — if `test_split` is empty or any Class 0
    invariant is violated (tech-spec v2 §6.3): a non-gold and/or non-Layer-B
    item (issue #11: an EvalItem(data_class="gold", layer="A") must not pass
    the acceptance gate; Layer A is transfer-calibrator-only, never
    sufficient for acceptance — CONTEXT.md "Three-layer evaluation"), a
    synthetic item tagged gold, an NFC-altering output, or a fixed-language
    output-language mismatch.

    `release_class` (tech-spec v2 §7): recorded verbatim on the returned
    report, not derived here — see `AcceptanceReport`'s docstring.

    `compute_metrics` (issue #34, issue #23 row D2): may return either a
    plain `dict[str, float]` (the original seam shape, kept so every
    existing caller/test in this module continues to work unchanged) or an
    `eval.metrics.MetricReport` (the typed shape `eval.metrics.
    compute_metrics` now produces). A `MetricReport`'s `.metrics` dict is
    unwrapped onto `AcceptanceReport.metrics` — the report type itself is
    not threaded through `AcceptanceReport`, since `AcceptanceReport` is
    this module's own return type and predates `MetricReport` by design
    (issue #29); unwrapping here is the seam's adapter point, not a
    structural coupling between the two types.

    `forgetting_axis_metrics`/`forgetting_axis_baseline`/
    `forgetting_axis_bootstrap_half_width` (tech-spec v2 §6.1): when all
    three are supplied, `flag_forgetting_regression` is run and its result
    plus the largest observed drop populate
    `AcceptanceReport.forgetting_axis_flagged`/`forgetting_axis_delta`. When
    any is omitted, no forgetting-axis comparison is made for this run and
    both fields take their no-comparison defaults (`False`/`None`) — an
    acceptance run is not required to carry a forgetting-axis comparison
    every time (e.g. the first arm run, with no prior "before" state).
    """
    if not test_split:
        raise AcceptanceGateError("test_split is empty; refusing to compute an acceptance report")

    violations = check_class_0_invariants(
        test_split,
        outputs=outputs,
        requested_languages=requested_languages,
        output_languages=output_languages,
    )
    if violations:
        raise AcceptanceGateError(
            "Class 0 invariant violation(s), refusing to compute an acceptance report "
            f"(tech-spec v2 §6.3): {violations}"
        )

    raw_metrics = compute_metrics(test_split, language)
    metrics = raw_metrics.metrics if isinstance(raw_metrics, MetricReport) else raw_metrics

    forgetting_axis_flagged = False
    forgetting_axis_delta: float | None = None
    if (
        forgetting_axis_metrics is not None
        and forgetting_axis_baseline is not None
        and forgetting_axis_bootstrap_half_width is not None
    ):
        forgetting_axis_flagged = flag_forgetting_regression(
            forgetting_axis_baseline,
            forgetting_axis_metrics,
            bootstrap_half_width=forgetting_axis_bootstrap_half_width,
        )
        shared_metrics = set(forgetting_axis_baseline) & set(forgetting_axis_metrics)
        if shared_metrics:
            forgetting_axis_delta = max(
                forgetting_axis_baseline[name] - forgetting_axis_metrics[name]
                for name in shared_metrics
            )

    return AcceptanceReport(
        language=language,
        metrics=metrics,
        item_count=len(test_split),
        release_class=release_class,
        forgetting_axis_flagged=forgetting_axis_flagged,
        forgetting_axis_delta=forgetting_axis_delta,
    )


def compute_calibration_report(
    language: LanguageTag,
    test_split: list[EvalItem],
    *,
    compute_metrics: Callable[[list[EvalItem], LanguageTag], dict[str, float]],
) -> CalibrationReport:
    """Layer-A calibration path: no gold-only gate, no Class 0 checks (those
    are acceptance-gate-specific — tech-spec v2 §6.3's invariant list names
    "a non-gold item in an acceptance split", not a calibration split).
    tech-spec v2 §6: "reported, never final acceptance" — enforced by
    CalibrationReport's distinct type, not by any restriction on which
    data_class values may appear here."""
    metrics = compute_metrics(test_split, language)
    return CalibrationReport(language=language, metrics=metrics, item_count=len(test_split))
