"""Tests for src/tracking/ — run-metadata record + injectable persist seam
(tech-spec §10).

Tests only external behavior: constructing RunMetadata fixtures, calling
record_run() with a test-double persist callable, asserting on call
count/args or the raised RunMetadataError. No real MLflow/DVC backend
exists — persist is always a test double here.
"""

from datetime import datetime, timedelta

import pytest

from tracking import (
    RunMetadata,
    RunMetadataError,
    new_run_id,
    record_run,
)

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _metadata(**overrides: object) -> RunMetadata:
    defaults: dict[str, object] = dict(
        run_id="run-001",
        stage="bakeoff",
        language="frc",
        config={"candidate_id": "mistral-7b-v0.3"},
        git_commit_sha="abc1234",
        started_at=_T0,
        completed_at=_T0 + timedelta(minutes=30),
        artifact_refs=["reports/bakeoff-run-001.json"],
    )
    defaults.update(overrides)
    return RunMetadata(**defaults)  # type: ignore[arg-type]


# --- record_run: success path --------------------------------------------------


def test_record_run_calls_persist_exactly_once_with_metadata() -> None:
    calls: list[RunMetadata] = []
    metadata = _metadata()
    record_run(metadata, persist=calls.append)
    assert calls == [metadata]


def test_record_run_returns_none() -> None:
    metadata = _metadata()
    result = record_run(metadata, persist=lambda m: None)
    assert result is None


# --- record_run: validation -----------------------------------------------------


def test_empty_run_id_raises_before_persist_is_called() -> None:
    call_count = {"n": 0}

    def persist(m: RunMetadata) -> None:
        call_count["n"] += 1

    metadata = _metadata(run_id="")
    with pytest.raises(RunMetadataError):
        record_run(metadata, persist=persist)
    assert call_count["n"] == 0


def test_completed_before_started_raises_before_persist_is_called() -> None:
    call_count = {"n": 0}

    def persist(m: RunMetadata) -> None:
        call_count["n"] += 1

    metadata = _metadata(started_at=_T0, completed_at=_T0 - timedelta(minutes=1))
    with pytest.raises(RunMetadataError):
        record_run(metadata, persist=persist)
    assert call_count["n"] == 0


def test_completed_equal_to_started_is_accepted() -> None:
    calls: list[RunMetadata] = []
    metadata = _metadata(started_at=_T0, completed_at=_T0)
    record_run(metadata, persist=calls.append)
    assert len(calls) == 1


# --- record_run: exception propagation from persist ------------------------------


def test_record_run_propagates_persist_exception_directly() -> None:
    def failing_persist(m: RunMetadata) -> None:
        raise RuntimeError("backend unavailable")

    metadata = _metadata()
    with pytest.raises(RuntimeError, match="backend unavailable"):
        record_run(metadata, persist=failing_persist)


# --- new_run_id -------------------------------------------------------------------


def test_new_run_id_returns_non_empty_string() -> None:
    run_id = new_run_id()
    assert isinstance(run_id, str)
    assert run_id


def test_new_run_id_is_unique_across_calls() -> None:
    assert new_run_id() != new_run_id()


# --- determinism --------------------------------------------------------------------


def test_record_run_is_deterministic() -> None:
    metadata = _metadata()
    calls: list[RunMetadata] = []
    record_run(metadata, persist=calls.append)
    record_run(metadata, persist=calls.append)
    assert calls == [metadata, metadata]
