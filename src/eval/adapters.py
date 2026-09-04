"""`eval.metrics.MetricReport` → `bakeoff.ScoreResult` adapter and the
`score`/`score_untuned_base` seams `bakeoff.run_bakeoff` requires (issue #43,
#23 row D2, #34 close-out "Wave 3: adapter onto `bakeoff.ScoreResult`").

`src/eval/metrics.py` computes a typed `MetricReport` (task, per-metric
values, a named headline float) but nothing before this module converts that
into the `(metric_name, value, higher_is_better)` triple `run_bakeoff`'s
`score`/`score_untuned_base` callables must return, so the bake-off harness
still could not be driven by real Layer-B metrics. This module is the seam:

- `HIGHER_IS_BETTER`: every metric name `compute_metrics` can emit, for
  every `eval.metrics.Task`, mapped to its comparison direction. An unknown
  name is a hard error (`UnknownMetricError`), never a silently-assumed
  default, since guessing the wrong direction would silently invert a
  bake-off winner.
- `score_result_from_report`/`raw_metrics_from_report`: pure `MetricReport`
  readers with no I/O.
- `make_score_seams`: builds the two callables `run_bakeoff` calls directly,
  closing over an injected `generate` callable (real model inference is out
  of scope for this ticket — see the issue's "Out of scope") and enforcing
  gold-vs-silver at the seam (utils-spec benchmark v2 §3 "Gold-vs-silver
  enforcement (runtime assertion, not a report note)"), independently of
  `src/eval`'s own `compute_acceptance_report` gate, since `run_bakeoff`
  never calls that function.

This ticket has no consumer this wave (the benchmark CLI's `bakeoff` mode is
Wave 4, issue #23 Wave 3 plan point 6); the proof here is against
`bakeoff.run_bakeoff` directly, driven by these seams with a stub
`fine_tune`/`run_red_team` (this module's own test file).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from bakeoff import Candidate, ScoreResult, TrainedArtifact
from data_contract import TargetLanguage
from eval import EvalItem
from eval.metrics import MetricReport, Task, compute_metrics

__all__ = [
    "HIGHER_IS_BETTER",
    "ScoreSeam",
    "UntunedBaseSeam",
    "UnknownMetricError",
    "GoldOnlyViolationError",
    "score_result_from_report",
    "raw_metrics_from_report",
    "make_score_seams",
]

# Every plain (non-per-cell) metric name `eval.metrics.compute_metrics` can
# emit, across all four `Task` values, mapped to its comparison direction
# (tech-spec v2 §6.1's per-task metrics table: chrF/chrF++/BLEU/accuracies
# are "higher is better"; WER/CER/TER/fertility are error rates, "lower is
# better"). Enumerated by reading `src/eval/metrics.py`'s three
# `_compute_*_metrics` helpers in full, not from memory:
#   `_compute_mt_metrics`        -> chrf_plus_plus, chrf, bleu
#   `_compute_normalize_metrics` -> token_error_rate, and (when every item
#                                   carries `reference_diplomatic`)
#                                   wer_diplomatic, wer_normalized,
#                                   cer_diplomatic, cer_normalized
#   `_compute_cell_accuracy_metrics` -> per_cell_accuracy (contrast/judge),
#                                   plus one dynamic `accuracy[<cell_id>]`
#                                   key per diff-catalog cell present in the
#                                   scored split (`cell_id` is data, not a
#                                   fixed name — matched by prefix below,
#                                   never enumerated here).
# `tokenizer_fertility` is a standalone function (tech-spec v2 §6.1 "tokens/
# word ... per candidate tokenizer") that `compute_metrics`/`MetricReport`
# never emits as a named key today (no `_compute_*_metrics` branch calls
# it), but is listed here anyway so a future wiring of it needs no table
# change, only a direction decision already made: fertility is an error/cost
# proxy (more tokens per word is worse for a low-resource tokenizer), so
# "lower is better".
HIGHER_IS_BETTER: Mapping[str, bool] = {
    "chrf_plus_plus": True,
    "chrf": True,
    "bleu": True,
    "token_error_rate": False,
    "wer_diplomatic": False,
    "wer_normalized": False,
    "cer_diplomatic": False,
    "cer_normalized": False,
    "per_cell_accuracy": True,
    "tokenizer_fertility": False,
}

# The dynamic per-cell key prefix `_compute_cell_accuracy_metrics` writes
# (`metrics.update({f"accuracy[{cell_id}]": ...})`) — matched by prefix
# since `cell_id` is diff-catalog data (tech-spec v2 §4), never enumerable
# ahead of time. Any key with this prefix is an accuracy, always
# "higher is better" (tech-spec v2 §6.1 "per-cell accuracy ... with Wilson
# CIs").
_PER_CELL_ACCURACY_PREFIX = "accuracy["


class UnknownMetricError(ValueError):
    """Raised by `score_result_from_report`/`_direction_for` when a metric
    name is not in `HIGHER_IS_BETTER` and does not match the per-cell
    accuracy prefix — never defaulted to a guessed direction, since a wrong
    guess would silently invert which arm wins a bake-off comparison."""


class GoldOnlyViolationError(ValueError):
    """Raised by a seam built with `make_score_seams` when the scoring set
    for a language contains a `synthetic=True` or non-`"gold"` `data_class`
    item (utils-spec benchmark v2 §3), or when a language has no items at
    all — in both cases before any `generate` call, matching the issue's
    "raises before any generation" requirement."""


def _direction_for(metric_name: str) -> bool:
    if metric_name in HIGHER_IS_BETTER:
        return HIGHER_IS_BETTER[metric_name]
    if metric_name.startswith(_PER_CELL_ACCURACY_PREFIX):
        return True
    raise UnknownMetricError(
        f"no comparison direction known for metric {metric_name!r}; "
        "add it to eval.adapters.HIGHER_IS_BETTER rather than assuming a default"
    )


def score_result_from_report(report: MetricReport) -> ScoreResult:
    """`report.headline_name`/`report.headline_value` as a `bakeoff.
    ScoreResult`, with `higher_is_better` looked up from `HIGHER_IS_BETTER`
    (raising `UnknownMetricError` rather than defaulting — see module
    docstring)."""
    return ScoreResult(
        metric_name=report.headline_name,
        value=report.headline_value,
        higher_is_better=_direction_for(report.headline_name),
    )


def raw_metrics_from_report(report: MetricReport) -> dict[str, float]:
    """Flatten a `MetricReport` into the `dict[str, float]` shape
    `run_bakeoff`'s `score` seam returns as its second tuple element
    (`raw_metrics`, accumulated onto `CandidateResult.raw_metrics`).

    Key scheme (pinned by `test_raw_metrics_key_scheme_pinned`):
    - every `report.metrics` key, verbatim;
    - for each entry in `report.per_cell`, two additional keys:
      `cell.<cell_id>.accuracy` and `cell.<cell_id>.n` (the per-cell Wilson
      CI bounds are not included — `run_bakeoff`'s `raw_metrics` seam type
      is `dict[str, float]`, a flat scalar map, and the CI bounds are
      already recoverable from `report.per_cell` for any caller working
      directly with the `MetricReport`, so this flattening only adds the
      two scalars `raw_metrics`'s float-map shape can hold without losing
      the cell's sample size, which the CI bounds alone do not carry).
    """
    flat: dict[str, float] = dict(report.metrics)
    for cell_id, cell in report.per_cell.items():
        flat[f"cell.{cell_id}.accuracy"] = cell.accuracy
        flat[f"cell.{cell_id}.n"] = float(cell.n)
    return flat


# The two seam types `bakeoff.run_bakeoff` requires verbatim (confirmed via
# `inspect.signature(bakeoff.run_bakeoff)` from within this worktree):
#   score: Callable[[TrainedArtifact, TargetLanguage], tuple[ScoreResult, dict[str, float]]]
#   score_untuned_base: Callable[[Candidate, TargetLanguage], ScoreResult]
ScoreSeam = Callable[[TrainedArtifact, TargetLanguage], tuple[ScoreResult, dict[str, float]]]
UntunedBaseSeam = Callable[[Candidate, TargetLanguage], ScoreResult]

# The `generate` callable this ticket injects rather than implements (real
# model inference is out of scope — see the issue's "Out of scope" and the
# module docstring). Takes the trained artifact (or, for the untuned-base
# arm, the `Candidate` itself) plus the ordered `EvalItem`s to score, and
# returns one hypothesis string per item, in the same order.
_Generate = Callable[["TrainedArtifact | Candidate", Sequence[EvalItem]], Sequence[str]]


def _require_gold_only(items: Sequence[EvalItem], *, language: TargetLanguage) -> None:
    """Gold-vs-silver enforcement at the seam (utils-spec benchmark v2 §3):
    raises `GoldOnlyViolationError` naming every offending item, or an empty
    split, before any `generate` call. Independent of `src/eval`'s own
    `compute_acceptance_report`/`AcceptanceGateError` gate, since
    `run_bakeoff` never calls that function — this seam is the only gate a
    bake-off run passes through."""
    if not items:
        raise GoldOnlyViolationError(
            f"language={language!r} has no items in the scoring set; refusing to score"
        )
    offending = [
        item.item_id
        for item in items
        if item.synthetic or item.data_class != "gold"
    ]
    if offending:
        raise GoldOnlyViolationError(
            f"language={language!r}: non-gold and/or synthetic item(s) in the scoring set, "
            f"refusing to generate or score (utils-spec benchmark v2 §3): {offending}"
        )


def make_score_seams(
    *,
    items_by_language: Mapping[TargetLanguage, Sequence[EvalItem]],
    task: Task,
    generate: _Generate,
) -> tuple[ScoreSeam, UntunedBaseSeam]:
    """Build the `score`/`score_untuned_base` callables `run_bakeoff`
    requires, both driven by real `eval.metrics.compute_metrics` and this
    module's report-to-`ScoreResult` adapter.

    `items_by_language` fixes each language's scoring set once, up front —
    `run_bakeoff` itself carries no notion of "the eval split", so this
    factory closes over it instead of inventing a new parameter on
    `run_bakeoff`. Both returned callables enforce gold-vs-silver (see
    `_require_gold_only`) before calling `generate`; a language absent from
    `items_by_language` is treated the same as an empty sequence (raises
    `GoldOnlyViolationError`, never a `KeyError`).

    The `score` seam calls `generate` with the `TrainedArtifact` itself (the
    trained candidate's own hypothesis generator); `score_untuned_base`
    calls it with the bare `Candidate` (there is no `TrainedArtifact` yet
    for the untuned base — tech-spec v2 §3.2 harness responsibility 3).
    Both then run `eval.metrics.compute_metrics(items, hypotheses, task)`
    and adapt the resulting `MetricReport` via `score_result_from_report`/
    `raw_metrics_from_report`.
    """

    def _scoring_items(language: TargetLanguage) -> Sequence[EvalItem]:
        items = items_by_language.get(language, ())
        _require_gold_only(items, language=language)
        return items

    def score(artifact: TrainedArtifact, language: TargetLanguage) -> tuple[ScoreResult, dict[str, float]]:
        items = _scoring_items(language)
        hypotheses = generate(artifact, items)
        report = compute_metrics(items, hypotheses, task, language=None)
        return score_result_from_report(report), raw_metrics_from_report(report)

    def score_untuned_base(candidate: Candidate, language: TargetLanguage) -> ScoreResult:
        items = _scoring_items(language)
        hypotheses = generate(candidate, items)
        report = compute_metrics(items, hypotheses, task, language=None)
        return score_result_from_report(report)

    return score, score_untuned_base
