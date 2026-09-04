"""Tests for src/eval/ — the silver-vs-gold acceptance gate (tech-spec v2
§6.4), the recorded model release class (§7), the forgetting-axis flag
(§6.1), and the Class 0 invariant hooks (§6.3).

Tests only external behavior: constructing EvalItem lists and calling
compute_acceptance_report()/compute_calibration_report()/
check_class_0_invariants()/flag_forgetting_regression(), asserting on the
returned report/value or raised error. Never asserts on internal iteration
order.
"""

import pytest

from data_contract import DataContractError, ModelReleaseClass
from eval import (
    AcceptanceGateError,
    AcceptanceReport,
    CalibrationReport,
    EvalItem,
    check_class_0_invariants,
    compute_acceptance_report,
    compute_calibration_report,
    flag_forgetting_regression,
)
from eval.metrics import MetricReport

_RELEASE_READY: ModelReleaseClass = "release_ready"


def _item(item_id: str, data_class: str, layer: str = "B", synthetic: bool = False) -> EvalItem:
    return EvalItem(
        item_id=item_id,
        language_tag="frc",
        data_class=data_class,  # type: ignore[arg-type]
        layer=layer,  # type: ignore[arg-type]
        synthetic=synthetic,
    )


def _metrics(items: list[EvalItem], language: str) -> dict[str, float]:
    return {"macro_f1": 0.9}


# --- compute_acceptance_report: the hard gate --------------------------------


def test_all_gold_split_produces_report() -> None:
    items = [_item("a", "gold"), _item("b", "gold")]
    report = compute_acceptance_report(
        "frc", items, compute_metrics=_metrics, release_class=_RELEASE_READY
    )
    assert isinstance(report, AcceptanceReport)
    assert report.metrics == {"macro_f1": 0.9}
    assert report.item_count == 2


def test_single_silver_item_raises_and_never_calls_compute_metrics() -> None:
    items = [_item("a", "gold"), _item("b", "silver")]
    call_count = {"n": 0}

    def metrics(items: list[EvalItem], language: str) -> dict[str, float]:
        call_count["n"] += 1
        return {}

    with pytest.raises(AcceptanceGateError):
        compute_acceptance_report(
            "frc", items, compute_metrics=metrics, release_class=_RELEASE_READY
        )
    assert call_count["n"] == 0


def test_gate_refuses_bronze_data_class() -> None:
    """issue #29: bronze (unreviewed harvested text) is new in v2's
    data_class set and must be caught by the same "not gold" gate check as
    silver/synthetic, alongside the existing single_silver_item test."""
    items = [_item("a", "gold"), _item("bronze-item", "bronze")]
    call_count = {"n": 0}

    def metrics(items: list[EvalItem], language: str) -> dict[str, float]:
        call_count["n"] += 1
        return {}

    with pytest.raises(AcceptanceGateError, match="bronze-item"):
        compute_acceptance_report(
            "frc", items, compute_metrics=metrics, release_class=_RELEASE_READY
        )
    assert call_count["n"] == 0


def test_error_message_names_the_offending_item() -> None:
    items = [_item("a", "gold"), _item("bad-item", "silver")]
    with pytest.raises(AcceptanceGateError, match="bad-item"):
        compute_acceptance_report(
            "frc", items, compute_metrics=_metrics, release_class=_RELEASE_READY
        )


def test_error_message_names_every_offending_item_not_just_first() -> None:
    items = [_item("silver-one", "silver"), _item("silver-two", "silver")]
    with pytest.raises(AcceptanceGateError) as exc_info:
        compute_acceptance_report(
            "frc", items, compute_metrics=_metrics, release_class=_RELEASE_READY
        )
    message = str(exc_info.value)
    assert "silver-one" in message
    assert "silver-two" in message


def test_empty_test_split_raises_before_calling_compute_metrics() -> None:
    call_count = {"n": 0}

    def metrics(items: list[EvalItem], language: str) -> dict[str, float]:
        call_count["n"] += 1
        return {}

    with pytest.raises((AcceptanceGateError, ValueError)):
        compute_acceptance_report(
            "frc", [], compute_metrics=metrics, release_class=_RELEASE_READY
        )
    assert call_count["n"] == 0


def test_gold_item_outside_layer_b_raises_and_never_calls_compute_metrics() -> None:
    """issue #11: EvalItem(data_class="gold", layer="A") must not pass the
    acceptance gate — gold alone is not sufficient, Layer B is required."""
    items = [_item("a", "gold", layer="A")]
    call_count = {"n": 0}

    def metrics(items: list[EvalItem], language: str) -> dict[str, float]:
        call_count["n"] += 1
        return {}

    with pytest.raises(AcceptanceGateError, match="a"):
        compute_acceptance_report(
            "frc", items, compute_metrics=metrics, release_class=_RELEASE_READY
        )
    assert call_count["n"] == 0


def test_acceptance_report_declares_layer_b() -> None:
    items = [_item("a", "gold"), _item("b", "gold")]
    report = compute_acceptance_report(
        "frc", items, compute_metrics=_metrics, release_class=_RELEASE_READY
    )
    assert report.layer == "B"


def test_compute_acceptance_report_raises_before_metrics_on_class_0_violation() -> None:
    """Extends the existing "never calls compute_metrics" pattern to the new
    Class 0 checks (issue #29 acceptance criteria): a synthetic item tagged
    gold data_class must refuse the report before compute_metrics runs, just
    like the non-gold/non-Layer-B check does."""
    items = [_item("a", "gold", synthetic=True)]
    call_count = {"n": 0}

    def metrics(items: list[EvalItem], language: str) -> dict[str, float]:
        call_count["n"] += 1
        return {}

    with pytest.raises(AcceptanceGateError, match="synthetic"):
        compute_acceptance_report(
            "frc", items, compute_metrics=metrics, release_class=_RELEASE_READY
        )
    assert call_count["n"] == 0


# --- AcceptanceReport.release_class -------------------------------------------


def test_acceptance_report_carries_release_class() -> None:
    items = [_item("a", "gold")]
    report = compute_acceptance_report(
        "frc", items, compute_metrics=_metrics, release_class="research_only"
    )
    assert report.release_class == "research_only"


# --- forgetting axis -----------------------------------------------------------


def test_flag_forgetting_regression_flags_when_drop_exceeds_2_points() -> None:
    baseline = {"chrf": 50.0}
    candidate = {"chrf": 47.5}  # drop of 2.5 > 2.0 absolute floor
    assert flag_forgetting_regression(baseline, candidate, bootstrap_half_width={"chrf": 0.1}) is True


def test_flag_forgetting_regression_flags_when_drop_exceeds_bootstrap_half_width() -> None:
    baseline = {"chrf": 50.0}
    candidate = {"chrf": 46.0}  # drop of 4.0, half_width 3.5 > 2.0 floor -> threshold 3.5
    assert (
        flag_forgetting_regression(baseline, candidate, bootstrap_half_width={"chrf": 3.5}) is True
    )


def test_flag_forgetting_regression_does_not_flag_within_tolerance() -> None:
    baseline = {"chrf": 50.0}
    candidate = {"chrf": 49.0}  # drop of 1.0, below both the 2.0 floor and half_width
    assert (
        flag_forgetting_regression(baseline, candidate, bootstrap_half_width={"chrf": 3.0}) is False
    )


def test_flag_forgetting_regression_ignores_metrics_not_shared() -> None:
    baseline = {"chrf": 50.0, "only_in_baseline": 10.0}
    candidate = {"chrf": 49.5, "only_in_candidate": 999.0}
    assert (
        flag_forgetting_regression(baseline, candidate, bootstrap_half_width={"chrf": 3.0}) is False
    )


def test_compute_acceptance_report_populates_forgetting_axis_when_supplied() -> None:
    items = [_item("a", "gold")]
    report = compute_acceptance_report(
        "frc",
        items,
        compute_metrics=_metrics,
        release_class=_RELEASE_READY,
        forgetting_axis_baseline={"chrf": 50.0},
        forgetting_axis_metrics={"chrf": 40.0},
        forgetting_axis_bootstrap_half_width={"chrf": 1.0},
    )
    assert report.forgetting_axis_flagged is True
    assert report.forgetting_axis_delta == pytest.approx(10.0)


def test_compute_acceptance_report_defaults_forgetting_axis_when_omitted() -> None:
    items = [_item("a", "gold")]
    report = compute_acceptance_report(
        "frc", items, compute_metrics=_metrics, release_class=_RELEASE_READY
    )
    assert report.forgetting_axis_flagged is False
    assert report.forgetting_axis_delta is None


# --- check_class_0_invariants ---------------------------------------------------


def test_check_class_0_invariants_flags_synthetic_in_gold_split() -> None:
    items = [_item("a", "gold", synthetic=True)]
    violations = check_class_0_invariants(items)
    assert len(violations) == 1
    assert "a" in violations[0]


def test_check_class_0_invariants_flags_nfc_decomposed_output() -> None:
    # "é" decomposed into "e" + combining acute accent is not NFC-normalized.
    decomposed = "é"
    items = [_item("a", "gold")]
    violations = check_class_0_invariants(items, outputs=[decomposed])
    assert len(violations) == 1
    assert "NFC" in violations[0] or "diacritic" in violations[0].lower()


def test_check_class_0_invariants_flags_output_language_mismatch() -> None:
    items = [_item("a", "gold")]
    violations = check_class_0_invariants(
        items,
        requested_languages=["frc"],
        output_languages=["fra"],
    )
    assert len(violations) == 1
    assert "language" in violations[0].lower()


def test_check_class_0_invariants_returns_empty_list_when_clean() -> None:
    items = [_item("a", "gold"), _item("b", "gold")]
    violations = check_class_0_invariants(
        items,
        outputs=["clean text"],
        requested_languages=["frc"],
        output_languages=["frc"],
    )
    assert violations == []


def test_check_class_0_invariants_flags_non_gold_item() -> None:
    items = [_item("silver-item", "silver")]
    violations = check_class_0_invariants(items)
    assert len(violations) == 1
    assert "silver-item" in violations[0]


# --- compute_calibration_report: no gate --------------------------------------


def test_calibration_report_succeeds_on_mixed_data_class_split() -> None:
    items = [_item("a", "gold", layer="A"), _item("b", "silver", layer="A")]
    report = compute_calibration_report("frc", items, compute_metrics=_metrics)
    assert isinstance(report, CalibrationReport)
    assert report.metrics == {"macro_f1": 0.9}


def test_calibration_report_succeeds_on_silver_only_split() -> None:
    items = [_item("a", "silver", layer="A"), _item("b", "silver", layer="A")]
    report = compute_calibration_report("frc", items, compute_metrics=_metrics)
    assert isinstance(report, CalibrationReport)


def test_calibration_report_succeeds_on_bronze_data_class() -> None:
    items = [_item("a", "bronze", layer="A")]
    report = compute_calibration_report("frc", items, compute_metrics=_metrics)
    assert isinstance(report, CalibrationReport)


def test_calibration_report_type_is_distinct_from_acceptance_report() -> None:
    # mypy proves `CalibrationReport is not AcceptanceReport` always True
    # (comparison-overlap) since the two share no subtype relationship;
    # routing through a plain object comparison keeps the intent (two
    # distinct types) as a real runtime check instead of a tautology mypy
    # would otherwise flag.
    types: list[object] = [CalibrationReport, AcceptanceReport]
    assert types[0] is not types[1]
    gold_items = [_item("a", "gold")]
    acceptance = compute_acceptance_report(
        "frc", gold_items, compute_metrics=_metrics, release_class=_RELEASE_READY
    )
    calibration = compute_calibration_report("frc", gold_items, compute_metrics=_metrics)
    assert not isinstance(acceptance, CalibrationReport)
    assert not isinstance(calibration, AcceptanceReport)


def test_calibration_report_declares_layer_a_by_default() -> None:
    items = [_item("a", "silver", layer="A")]
    report = compute_calibration_report("frc", items, compute_metrics=_metrics)
    assert report.layer == "A"


# --- EvalItem: closed-enum validation (issue #11 / #29) -------------------------


def test_eval_item_data_class_field_replaces_tier() -> None:
    """issue #29: constructing EvalItem(data_class=..., ...) succeeds; the
    old tier= kwarg raises TypeError (hard rename, no compatibility
    alias)."""
    item = EvalItem(item_id="a", language_tag="frc", data_class="gold", layer="B")
    assert item.data_class == "gold"
    with pytest.raises(TypeError):
        EvalItem(item_id="a", language_tag="frc", tier="gold", layer="B")  # type: ignore[call-arg]


def test_eval_item_rejects_data_class_outside_closed_set() -> None:
    with pytest.raises(DataContractError):
        EvalItem(item_id="a", language_tag="frc", data_class="not_a_real_class", layer="B")  # type: ignore[arg-type]


def test_eval_item_rejects_layer_outside_closed_set() -> None:
    with pytest.raises(DataContractError):
        EvalItem(item_id="a", language_tag="frc", data_class="gold", layer="D")  # type: ignore[arg-type]


def test_eval_item_accepts_synthetic_data_class() -> None:
    """data_class includes "synthetic" (issue #11's single-owner partition,
    carried forward under the v2 name) — an EvalItem must be able to declare
    it, even though the acceptance gate still only lets "gold" through."""
    item = EvalItem(item_id="a", language_tag="frc", data_class="synthetic", layer="A", synthetic=True)
    assert item.data_class == "synthetic"


def test_eval_item_accepts_bronze_data_class() -> None:
    item = EvalItem(item_id="a", language_tag="frc", data_class="bronze", layer="A")
    assert item.data_class == "bronze"


def test_eval_item_synthetic_defaults_false() -> None:
    item = EvalItem(item_id="a", language_tag="frc", data_class="gold", layer="B")
    assert item.synthetic is False


def test_eval_item_gains_source_text_and_references() -> None:
    """issue #34, issue #23 row D2: EvalItem must carry text/reference
    fields for the metric implementations in eval.metrics to score against.
    All four new fields default to None, so a caller with no text (e.g. an
    LID-only item) is unaffected."""
    item = EvalItem(item_id="a", language_tag="frc", data_class="gold", layer="B")
    assert item.source_text is None
    assert item.reference is None
    assert item.reference_diplomatic is None
    assert item.cell_id is None

    populated = EvalItem(
        item_id="b",
        language_tag="frc",
        data_class="gold",
        layer="B",
        source_text="ils mangeont",
        reference="ils mangent",
        reference_diplomatic="ils mangeont",
        cell_id="VERB-001",
    )
    assert populated.source_text == "ils mangeont"
    assert populated.reference == "ils mangent"
    assert populated.reference_diplomatic == "ils mangeont"
    assert populated.cell_id == "VERB-001"


# --- determinism ---------------------------------------------------------------


def test_acceptance_report_is_deterministic() -> None:
    items = [_item("a", "gold"), _item("b", "gold")]
    first = compute_acceptance_report(
        "frc", items, compute_metrics=_metrics, release_class=_RELEASE_READY
    )
    second = compute_acceptance_report(
        "frc", items, compute_metrics=_metrics, release_class=_RELEASE_READY
    )
    assert first == second


def test_calibration_report_is_deterministic() -> None:
    items = [_item("a", "silver", layer="A")]
    first = compute_calibration_report("frc", items, compute_metrics=_metrics)
    second = compute_calibration_report("frc", items, compute_metrics=_metrics)
    assert first == second


# --- compute_acceptance_report accepts a MetricReport (issue #34) --------------


def test_compute_acceptance_report_accepts_metric_report() -> None:
    """issue #34: compute_metrics may now return eval.metrics.MetricReport
    instead of a bare dict; compute_acceptance_report must unwrap its
    `.metrics` dict onto AcceptanceReport.metrics rather than requiring
    every caller to unwrap it first. The bare-dict path (`_metrics` above)
    stays supported for every other test in this file."""

    def metrics_report(items: list[EvalItem], language: str) -> MetricReport:
        return MetricReport(
            task="mt",
            metrics={"chrf_plus_plus": 87.5, "bleu": 60.0},
            headline_name="chrf_plus_plus",
            headline_value=87.5,
        )

    items = [_item("a", "gold"), _item("b", "gold")]
    report = compute_acceptance_report(
        "frc", items, compute_metrics=metrics_report, release_class=_RELEASE_READY
    )
    assert isinstance(report, AcceptanceReport)
    assert report.metrics == {"chrf_plus_plus": 87.5, "bleu": 60.0}
    assert report.item_count == 2
