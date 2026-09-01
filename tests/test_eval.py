"""Tests for src/eval/ — the silver-vs-gold acceptance gate (tech-spec §6.4).

Tests only external behavior: constructing EvalItem lists and calling
compute_acceptance_report()/compute_calibration_report(), asserting on the
returned report or raised error. Never asserts on internal iteration order.
"""

import pytest

from eval import (
    AcceptanceGateError,
    AcceptanceReport,
    CalibrationReport,
    EvalItem,
    compute_acceptance_report,
    compute_calibration_report,
)


def _item(item_id: str, tier: str, layer: str = "B") -> EvalItem:
    return EvalItem(item_id=item_id, language_tag="frc", tier=tier, layer=layer)  # type: ignore[arg-type]


def _metrics(items: list[EvalItem], language: str) -> dict[str, float]:
    return {"macro_f1": 0.9}


# --- compute_acceptance_report: the hard gate --------------------------------


def test_all_gold_split_produces_report() -> None:
    items = [_item("a", "gold"), _item("b", "gold")]
    report = compute_acceptance_report("frc", items, compute_metrics=_metrics)
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
        compute_acceptance_report("frc", items, compute_metrics=metrics)
    assert call_count["n"] == 0


def test_error_message_names_the_offending_item() -> None:
    items = [_item("a", "gold"), _item("bad-item", "silver")]
    with pytest.raises(AcceptanceGateError, match="bad-item"):
        compute_acceptance_report("frc", items, compute_metrics=_metrics)


def test_error_message_names_every_offending_item_not_just_first() -> None:
    items = [_item("silver-one", "silver"), _item("silver-two", "silver")]
    with pytest.raises(AcceptanceGateError) as exc_info:
        compute_acceptance_report("frc", items, compute_metrics=_metrics)
    message = str(exc_info.value)
    assert "silver-one" in message
    assert "silver-two" in message


def test_empty_test_split_raises_before_calling_compute_metrics() -> None:
    call_count = {"n": 0}

    def metrics(items: list[EvalItem], language: str) -> dict[str, float]:
        call_count["n"] += 1
        return {}

    with pytest.raises((AcceptanceGateError, ValueError)):
        compute_acceptance_report("frc", [], compute_metrics=metrics)
    assert call_count["n"] == 0


# --- compute_calibration_report: no gate --------------------------------------


def test_calibration_report_succeeds_on_mixed_tier_split() -> None:
    items = [_item("a", "gold", layer="A"), _item("b", "silver", layer="A")]
    report = compute_calibration_report("frc", items, compute_metrics=_metrics)
    assert isinstance(report, CalibrationReport)
    assert report.metrics == {"macro_f1": 0.9}


def test_calibration_report_succeeds_on_silver_only_split() -> None:
    items = [_item("a", "silver", layer="A"), _item("b", "silver", layer="A")]
    report = compute_calibration_report("frc", items, compute_metrics=_metrics)
    assert isinstance(report, CalibrationReport)


def test_calibration_report_type_is_distinct_from_acceptance_report() -> None:
    assert CalibrationReport is not AcceptanceReport
    gold_items = [_item("a", "gold")]
    acceptance = compute_acceptance_report("frc", gold_items, compute_metrics=_metrics)
    calibration = compute_calibration_report("frc", gold_items, compute_metrics=_metrics)
    assert not isinstance(acceptance, CalibrationReport)
    assert not isinstance(calibration, AcceptanceReport)


# --- determinism ---------------------------------------------------------------


def test_acceptance_report_is_deterministic() -> None:
    items = [_item("a", "gold"), _item("b", "gold")]
    first = compute_acceptance_report("frc", items, compute_metrics=_metrics)
    second = compute_acceptance_report("frc", items, compute_metrics=_metrics)
    assert first == second


def test_calibration_report_is_deterministic() -> None:
    items = [_item("a", "silver", layer="A")]
    first = compute_calibration_report("frc", items, compute_metrics=_metrics)
    second = compute_calibration_report("frc", items, compute_metrics=_metrics)
    assert first == second
