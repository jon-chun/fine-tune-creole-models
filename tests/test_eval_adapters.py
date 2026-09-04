"""Tests for src/eval/adapters.py — the `MetricReport` -> `bakeoff.
ScoreResult` adapter and the `score`/`score_untuned_base` seams `bakeoff.
run_bakeoff` requires (issue #43, #23 row D2).

This ticket has no consumer this wave (the benchmark CLI's `bakeoff` mode is
Wave 4), so the proof that the seams actually fit `run_bakeoff` is here:
driving the real harness with a stub `fine_tune` and a stub `run_red_team`.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import get_args

import pytest

from bakeoff import (
    BakeoffDefaults,
    BakeoffConfig,
    Candidate,
    FrcLanguageConfig,
    LouLanguageConfig,
    RedTeamCellResult,
    RedTeamVerdict,
    ScoreResult,
    TrainedArtifact,
    run_bakeoff,
)
from data_contract import TargetLanguage
from eval import EvalItem
from eval.adapters import (
    HIGHER_IS_BETTER,
    GoldOnlyViolationError,
    UnknownMetricError,
    make_score_seams,
    raw_metrics_from_report,
    score_result_from_report,
)
from eval.metrics import MetricReport, Task, compute_metrics, headline_metric


# --- fixtures shared across the run_bakeoff-driving tests --------------------


def _candidate(id_: str) -> Candidate:
    return Candidate(
        id=id_,
        hf_repo=f"org/{id_}",
        license="apache-2.0",
        size_b=None,
        size_m=None,
        instruct=True,
        optional=False,
        alt=None,
        base=None,
        lang_token=None,
        note=None,
    )


def _config_with(candidates: list[Candidate]) -> BakeoffConfig:
    defaults = BakeoffDefaults(
        instruct=True,
        seeds=1,
        forgetting_axis="required",
        release_licenses=("apache-2.0", "mit"),
        nf4_min_size_b=20,
    )
    frc = FrcLanguageConfig(control="untuned_base", arms=tuple(candidates))
    lou = LouLanguageConfig(
        control="untuned_base",
        generative_arms=(_candidate("lou-g1"), _candidate("lou-g2")),
        mt_arms=(_candidate("lou-m1"), _candidate("lou-m2"), _candidate("lou-m3")),
        prerequisite="x",
        decision_rule="y",
    )
    return BakeoffConfig(schema_version="2.0.0", defaults=defaults, frc=frc, lou=lou)


def _gold_item(item_id: str, reference: str) -> EvalItem:
    return EvalItem(
        item_id=item_id,
        language_tag="frc",
        data_class="gold",
        layer="B",
        synthetic=False,
        reference=reference,
    )


def _passing_red_team() -> RedTeamVerdict:
    cell = RedTeamCellResult(
        cell_id="VERB-001",
        base_rate=0.8,
        tuned_rate=0.1,
        wilson_95=(0.05, 0.15),
        mcnemar_p=0.001,
        gate_class=3,
        class_assigned=3,
    )
    return RedTeamVerdict(probe_set_version="v-test", cells={"VERB-001": cell})


def _stub_fine_tune(candidate: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
    return TrainedArtifact(
        candidate_id=candidate.id,
        adapter_ref=f"adapter://{candidate.id}/{seed}",
        run_id=f"run-{candidate.id}-{seed}",
        hyperparameters_digest="digest-fixed",
        seed=seed,
    )


def _stub_run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
    return _passing_red_team()


# --- test_higher_is_better_covers_every_metric_compute_metrics_emits --------


@pytest.mark.parametrize("task", get_args(Task))
def test_higher_is_better_covers_every_metric_compute_metrics_emits(task: Task) -> None:
    # Drive compute_metrics for every task family with a fixture that
    # exercises every optional branch (wer_cer_decomposed needs
    # reference_diplomatic on every item; contrast/judge needs cell_id) and
    # confirm every emitted metric name resolves a direction without
    # UnknownMetricError.
    if task == "mt":
        items = [_item_for_task(task, "a", reference="bonjour")]
        hypotheses = ["bonjour"]
    elif task == "normalize":
        items = [
            EvalItem(
                item_id="a",
                language_tag="frc",
                data_class="gold",
                layer="B",
                reference="normalized text",
                reference_diplomatic="diplomatic text",
            )
        ]
        hypotheses = ["normalized text"]
    else:  # contrast / judge
        items = [
            EvalItem(
                item_id="a",
                language_tag="frc",
                data_class="gold",
                layer="B",
                reference="expected",
                cell_id="VERB-001",
            )
        ]
        hypotheses = ["expected"]

    report = compute_metrics(items, hypotheses, task)
    for metric_name in report.metrics:
        # Must not raise.
        direction = HIGHER_IS_BETTER.get(metric_name)
        if direction is None:
            assert metric_name.startswith("accuracy["), (
                f"metric {metric_name!r} from task {task!r} has no direction "
                "in HIGHER_IS_BETTER and does not match the per-cell prefix"
            )


def _item_for_task(task: Task, item_id: str, *, reference: str) -> EvalItem:
    return EvalItem(
        item_id=item_id,
        language_tag="frc",
        data_class="gold",
        layer="B",
        reference=reference,
    )


# --- test_unknown_metric_name_raises -----------------------------------------


def test_unknown_metric_name_raises() -> None:
    bogus_report = MetricReport(
        task="mt",
        metrics={"totally_unknown_metric": 1.0},
        headline_name="totally_unknown_metric",
        headline_value=1.0,
    )
    with pytest.raises(UnknownMetricError):
        score_result_from_report(bogus_report)


# --- test_score_result_uses_headline_metric_and_direction -------------------


def test_score_result_uses_headline_metric_and_direction() -> None:
    items = [_item_for_task("mt", "a", reference="bonjour le monde")]
    hypotheses = ["bonjour le monde"]
    report = compute_metrics(items, hypotheses, "mt")
    result = score_result_from_report(report)
    assert result.metric_name == headline_metric("mt") == "chrf_plus_plus"
    assert result.value == pytest.approx(report.headline_value)
    assert result.higher_is_better is True


# --- test_lower_is_better_for_wer_cer_ter ------------------------------------


def test_lower_is_better_for_wer_cer_ter() -> None:
    for metric_name in ("token_error_rate", "wer_diplomatic", "wer_normalized", "cer_diplomatic", "cer_normalized"):
        assert HIGHER_IS_BETTER[metric_name] is False

    items = [
        EvalItem(
            item_id="a",
            language_tag="frc",
            data_class="gold",
            layer="B",
            reference="le chat noir",
            reference_diplomatic="le chat noir",
        )
    ]
    report = compute_metrics(items, ["le chat noir"], "normalize")
    result = score_result_from_report(report)
    assert result.metric_name == "token_error_rate"
    assert result.higher_is_better is False


# --- test_raw_metrics_key_scheme_pinned --------------------------------------


def test_raw_metrics_key_scheme_pinned() -> None:
    items = [
        EvalItem(item_id="a", language_tag="frc", data_class="gold", layer="B", reference="ok", cell_id="VERB-001"),
        EvalItem(item_id="b", language_tag="frc", data_class="gold", layer="B", reference="ok", cell_id="VERB-001"),
        EvalItem(item_id="c", language_tag="frc", data_class="gold", layer="B", reference="no", cell_id="NOUN-002"),
    ]
    hypotheses = ["ok", "wrong", "no"]
    report = compute_metrics(items, hypotheses, "contrast")

    flat = raw_metrics_from_report(report)

    # Every report.metrics key verbatim.
    for key, value in report.metrics.items():
        assert flat[key] == pytest.approx(value)

    # Plus cell.<cell_id>.accuracy and cell.<cell_id>.n per per_cell entry.
    assert flat["cell.VERB-001.accuracy"] == pytest.approx(0.5)  # 1/2 correct
    assert flat["cell.VERB-001.n"] == pytest.approx(2.0)
    assert flat["cell.NOUN-002.accuracy"] == pytest.approx(1.0)
    assert flat["cell.NOUN-002.n"] == pytest.approx(1.0)


# --- test_score_seams_match_run_bakeoff_parameter_types ----------------------


def test_score_seams_match_run_bakeoff_parameter_types() -> None:
    sig = inspect.signature(run_bakeoff)
    score_param = sig.parameters["score"]
    score_untuned_base_param = sig.parameters["score_untuned_base"]

    def generate(artifact_or_candidate: TrainedArtifact | Candidate, items: Sequence[EvalItem]) -> Sequence[str]:
        return []

    score, score_untuned_base = make_score_seams(
        items_by_language={"frc": [_gold_item("a", "ref")]},
        task="mt",
        generate=generate,
    )

    # Confirm the annotation strings agree on shape (parameter count and
    # names aren't runtime-checkable further than this without a static
    # type checker; mypy's own pass on this file is the real check that the
    # concrete callables satisfy run_bakeoff's Callable[...] parameter
    # types, since inspect.signature erases the exact Callable[[...], ...]
    # parameterization to a string annotation).
    assert "ScoreResult" in str(score_param.annotation)
    assert "ScoreResult" in str(score_untuned_base_param.annotation)
    assert callable(score)
    assert callable(score_untuned_base)


# --- test_run_bakeoff_driven_by_real_seams_with_stub_fine_tune --------------


def test_run_bakeoff_driven_by_real_seams_with_stub_fine_tune() -> None:
    candidates = [_candidate("winner"), _candidate("loser")]

    items_by_language: dict[TargetLanguage, list[EvalItem]] = {
        "frc": [
            _gold_item("a", "bonjour le monde"),
            _gold_item("b", "comment allez vous"),
        ]
    }

    def generate(artifact_or_candidate: TrainedArtifact | Candidate, items: Sequence[EvalItem]) -> Sequence[str]:
        # "winner"'s TrainedArtifact/Candidate id perfectly matches every
        # reference; "loser"'s never matches, so chrF++ separates them.
        candidate_id = getattr(artifact_or_candidate, "candidate_id", None) or getattr(
            artifact_or_candidate, "id"
        )
        if candidate_id.startswith("winner"):
            return [item.reference for item in items if item.reference is not None]
        return ["completely unrelated text"] * len(items)

    score, score_untuned_base = make_score_seams(
        items_by_language=items_by_language, task="mt", generate=generate
    )

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters={"rank": 8},
        split_id="split-test",
        seeds=(1,),
        fine_tune=_stub_fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=_stub_run_red_team,
    )

    assert result.winner_candidate_id == "winner"
    for candidate_result in result.results:
        assert candidate_result.score is not None
        assert candidate_result.score.metric_name == headline_metric("mt")
        assert candidate_result.untuned_base_score is not None
        assert candidate_result.untuned_base_score.metric_name == headline_metric("mt")
        report = compute_metrics(
            items_by_language["frc"],
            [item.reference for item in items_by_language["frc"] if item.reference is not None],
            "mt",
        )
        assert set(report.metrics) <= set(candidate_result.raw_metrics)


# --- test_untuned_base_scored_once_per_candidate -----------------------------


def test_untuned_base_scored_once_per_candidate() -> None:
    candidates = [_candidate("a"), _candidate("b")]
    call_count = {"score_untuned_base": 0}

    items_by_language: dict[TargetLanguage, list[EvalItem]] = {"frc": [_gold_item("a", "hello world")]}

    def generate(artifact_or_candidate: TrainedArtifact | Candidate, items: Sequence[EvalItem]) -> Sequence[str]:
        if isinstance(artifact_or_candidate, Candidate):
            call_count["score_untuned_base"] += 1
        return [item.reference for item in items if item.reference is not None]

    score, score_untuned_base = make_score_seams(
        items_by_language=items_by_language, task="mt", generate=generate
    )

    run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters={"rank": 8},
        split_id="split-test",
        seeds=(1, 2, 3),
        fine_tune=_stub_fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=_stub_run_red_team,
    )

    # 2 candidates -> score_untuned_base called exactly once each, regardless
    # of the 3-seed loop (tech-spec v2 §3.2 harness responsibility 3).
    assert call_count["score_untuned_base"] == 2


# --- test_seam_refuses_synthetic_or_non_gold_items --------------------------


def test_seam_refuses_synthetic_or_non_gold_items() -> None:
    generate_called = {"count": 0}

    def generate(artifact_or_candidate: TrainedArtifact | Candidate, items: Sequence[EvalItem]) -> Sequence[str]:
        generate_called["count"] += 1
        return ["x" for _ in items]

    silver_item = EvalItem(
        item_id="silver-1", language_tag="frc", data_class="silver", layer="B", reference="x"
    )
    synthetic_gold_item = EvalItem(
        item_id="synth-1",
        language_tag="frc",
        data_class="gold",
        layer="B",
        synthetic=True,
        reference="x",
    )

    for bad_item in (silver_item, synthetic_gold_item):
        score, score_untuned_base = make_score_seams(
            items_by_language={"frc": [bad_item]}, task="mt", generate=generate
        )
        with pytest.raises(GoldOnlyViolationError):
            score_untuned_base(_candidate("a"), "frc")
        assert generate_called["count"] == 0  # never called

        artifact = _stub_fine_tune(_candidate("a"), {}, "split", 1)
        with pytest.raises(GoldOnlyViolationError):
            score(artifact, "frc")
        assert generate_called["count"] == 0  # never called


# --- test_seam_refuses_language_without_items --------------------------------


def test_seam_refuses_language_without_items() -> None:
    generate_called = {"count": 0}

    def generate(artifact_or_candidate: TrainedArtifact | Candidate, items: Sequence[EvalItem]) -> Sequence[str]:
        generate_called["count"] += 1
        return []

    # "frc" present but empty, and "lou" entirely absent from the mapping —
    # both must raise, neither may call generate.
    score, score_untuned_base = make_score_seams(
        items_by_language={"frc": []}, task="mt", generate=generate
    )
    with pytest.raises(GoldOnlyViolationError):
        score_untuned_base(_candidate("a"), "frc")

    score2, score_untuned_base2 = make_score_seams(
        items_by_language={"frc": [_gold_item("a", "x")]}, task="mt", generate=generate
    )
    with pytest.raises(GoldOnlyViolationError):
        score_untuned_base2(_candidate("a"), "lou")

    assert generate_called["count"] == 0
