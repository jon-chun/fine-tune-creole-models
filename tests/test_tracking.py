"""Tests for src/tracking/ — run-metadata record + injectable persist seam
(tech-spec §10) and src/tracking/backends.py — the local file backend,
MLflow payload flattening, and the git SHA reader (issue #37, ADR 0016).

Tests only external behavior: constructing RunMetadata fixtures, calling
record_run() with a test-double or LocalFileBackend persist callable, and
asserting on call count/args, raised errors, or on-disk/returned payload
shape. No real MLflow/DVC backend exists — persist is a local file store or
a test double here; a real MLflow client is never imported or called.
"""

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tracking import (
    RunMetadata,
    RunMetadataError,
    new_run_id,
    record_run,
    to_hashable_config,
    warnings_for_run,
)
from tracking.backends import (
    LocalFileBackend,
    TrackingBackendError,
    flatten_config,
    read_git_commit_sha,
    run_metadata_from_dict,
    run_metadata_to_dict,
    to_mlflow_payload,
)

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _metadata(**overrides: object) -> RunMetadata:
    defaults: dict[str, object] = dict(
        run_id="run-001",
        stage="bakeoff",
        language="frc",
        config=to_hashable_config({"candidate_id": "mistral-7b-v0.3"}),
        git_commit_sha="abc1234",
        started_at=_T0,
        completed_at=_T0 + timedelta(minutes=30),
        artifact_refs=("reports/bakeoff-run-001.json",),
        seed=42,
        split_id="split-2026-01-01",
        lock_hash="deadbeef",
        tree_dirty=False,
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
    # record_run is typed to return None (mypy would flag an assignment from
    # it as func-returns-value); the contract we're asserting is that calling
    # it raises nothing and yields no value to use, not a runtime None check.
    record_run(metadata, persist=lambda m: None)


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


# --- hashability (RunMetadata is frozen=True, slots=True) ------------------------


def test_run_metadata_is_hashable() -> None:
    metadata = _metadata()
    assert isinstance(hash(metadata), int)


def test_run_metadata_with_nested_config_is_hashable() -> None:
    metadata = _metadata(
        config=to_hashable_config(
            {"candidate_id": "mistral-7b-v0.3", "nested": {"a": 1}, "items": [1, 2, 3]}
        )
    )
    assert isinstance(hash(metadata), int)


def test_run_metadata_with_new_optional_fields_is_still_hashable() -> None:
    metadata = _metadata(
        image_digest="sha256:abcd1234",
        manifest_sha256="sha256:deadbeef",
        gpu_hours=4.5,
        usd=12.34,
        instance="H100-80GB",
    )
    assert isinstance(hash(metadata), int)


# --- Stage extension: preprocess/augment (tech-spec §11 end-to-end test) --------


def test_preprocess_stage_is_valid() -> None:
    metadata = _metadata(stage="preprocess")
    assert metadata.stage == "preprocess"


def test_augment_stage_is_valid() -> None:
    metadata = _metadata(stage="augment")
    assert metadata.stage == "augment"


def test_speech_eval_stage_is_valid() -> None:
    metadata = _metadata(stage="speech_eval")
    assert metadata.stage == "speech_eval"


def test_align_stage_is_valid() -> None:
    metadata = _metadata(stage="align")
    assert metadata.stage == "align"


def test_hitl_stage_is_valid() -> None:
    metadata = _metadata(stage="hitl")
    assert metadata.stage == "hitl"


# --- to_hashable_config -----------------------------------------------------------


def test_to_hashable_config_converts_nested_dict_and_list() -> None:
    frozen = to_hashable_config({"a": 1, "nested": {"b": 2}, "items": [1, 2]})
    assert frozen["a"] == 1
    assert frozen["nested"]["b"] == 2  # type: ignore[index]
    assert frozen["items"] == (1, 2)


# --- warnings_for_run --------------------------------------------------------------


def test_warnings_for_run_carries_sequence_into_tuple() -> None:
    result = warnings_for_run(["rank=64 is outside the documented range"])
    assert result == ("rank=64 is outside the documented range",)


def test_warnings_for_run_empty_sequence_is_empty_tuple() -> None:
    assert warnings_for_run([]) == ()


def test_run_metadata_default_warnings_is_empty_tuple() -> None:
    metadata = _metadata()
    assert metadata.warnings == ()


# --- new optional fields: image_digest/manifest_sha256/gpu_hours/usd/instance ----


def test_run_metadata_optional_fields_default_to_none_when_omitted() -> None:
    metadata = _metadata()
    assert metadata.image_digest is None
    assert metadata.manifest_sha256 is None
    assert metadata.gpu_hours is None
    assert metadata.usd is None
    assert metadata.instance is None


def test_run_metadata_carries_gpu_hours_usd_instance_when_provided() -> None:
    metadata = _metadata(gpu_hours=4.5, usd=12.34, instance="H100-80GB")
    assert metadata.gpu_hours == 4.5
    assert metadata.usd == 12.34
    assert metadata.instance == "H100-80GB"


def test_run_metadata_carries_image_digest_and_manifest_sha256() -> None:
    metadata = _metadata(
        image_digest="sha256:abcd1234", manifest_sha256="sha256:deadbeef"
    )
    assert metadata.image_digest == "sha256:abcd1234"
    assert metadata.manifest_sha256 == "sha256:deadbeef"


# --- stage validation (issue #15) --------------------------------------------


def test_run_metadata_rejects_invalid_stage() -> None:
    with pytest.raises(RunMetadataError):
        _metadata(stage="deploy")


# --- backends.run_metadata_to_dict / run_metadata_from_dict ----------------------


def test_run_json_roundtrips_run_metadata_including_datetime() -> None:
    metadata = _metadata(
        image_digest="sha256:abcd1234",
        manifest_sha256="sha256:deadbeef",
        gpu_hours=4.5,
        usd=12.34,
        instance="H100-80GB",
        warnings=("rank=64 is outside the documented range",),
    )
    payload = run_metadata_to_dict(metadata)
    # Must be JSON-serializable as-is (datetimes/FrozenMapping/tuples are the
    # three non-JSON-native shapes RunMetadata carries).
    json_text = json.dumps(payload)
    restored = run_metadata_from_dict(json.loads(json_text))
    assert restored == metadata


def test_run_metadata_to_dict_uses_iso8601_for_datetimes() -> None:
    metadata = _metadata()
    payload = run_metadata_to_dict(metadata)
    assert payload["started_at"] == _T0.isoformat()
    assert payload["completed_at"] == (_T0 + timedelta(minutes=30)).isoformat()


# --- LocalFileBackend -------------------------------------------------------------


def test_local_backend_writes_one_json_per_run_and_index_line(tmp_path: Path) -> None:
    backend = LocalFileBackend(tmp_path)
    metadata = _metadata()
    backend.persist(metadata)

    run_json_path = tmp_path / "runs" / f"{metadata.run_id}.json"
    index_path = tmp_path / "runs" / "index.jsonl"
    assert run_json_path.is_file()
    assert index_path.is_file()

    on_disk = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert run_metadata_from_dict(on_disk) == metadata

    index_lines = index_path.read_text(encoding="utf-8").splitlines()
    assert len(index_lines) == 1
    assert run_metadata_from_dict(json.loads(index_lines[0])) == metadata


def test_local_backend_appends_index_line_per_persist_call(tmp_path: Path) -> None:
    backend = LocalFileBackend(tmp_path)
    backend.persist(_metadata(run_id="run-001"))
    backend.persist(_metadata(run_id="run-002"))

    index_path = tmp_path / "runs" / "index.jsonl"
    index_lines = index_path.read_text(encoding="utf-8").splitlines()
    assert len(index_lines) == 2
    assert (tmp_path / "runs" / "run-001.json").is_file()
    assert (tmp_path / "runs" / "run-002.json").is_file()


def test_record_run_calls_persist_exactly_once_with_local_backend(tmp_path: Path) -> None:
    backend = LocalFileBackend(tmp_path)
    calls: list[RunMetadata] = []
    original_persist = backend.persist

    def counting_persist(metadata: RunMetadata) -> None:
        calls.append(metadata)
        original_persist(metadata)

    metadata = _metadata()
    record_run(metadata, persist=counting_persist)

    assert calls == [metadata]
    assert (tmp_path / "runs" / f"{metadata.run_id}.json").is_file()


# --- flatten_config -----------------------------------------------------------------


def test_flatten_config_dotted_keys_and_scalar_values() -> None:
    config = to_hashable_config(
        {
            "candidate_id": "mistral-7b-v0.3",
            "lora": {"rank": 16, "alpha": 32},
            "seeds": [1, 2, 3],
            "use_bf16": True,
        }
    )
    flat = flatten_config(config)
    assert flat["candidate_id"] == "mistral-7b-v0.3"
    assert flat["lora.rank"] == 16
    assert flat["lora.alpha"] == 32
    assert flat["seeds.0"] == 1
    assert flat["seeds.1"] == 2
    assert flat["seeds.2"] == 3
    assert flat["use_bf16"] is True


def test_flatten_config_stringifies_non_scalar_leaf() -> None:
    config = to_hashable_config({"note": None})
    flat = flatten_config(config)
    assert flat["note"] == "None"


# --- to_mlflow_payload ---------------------------------------------------------------


def test_to_mlflow_payload_splits_params_metrics_tags() -> None:
    metadata = _metadata(gpu_hours=4.5, usd=12.34, warnings=("w1", "w2"))
    params, metrics, tags = to_mlflow_payload(metadata)

    assert params["run_id"] == metadata.run_id
    assert params["stage"] == "bakeoff"
    assert params["candidate_id"] == "mistral-7b-v0.3"
    assert metrics["gpu_hours"] == 4.5
    assert metrics["usd"] == 12.34
    assert tags["tree_dirty"] == "False"
    assert tags["warnings"] == "w1; w2"


def test_to_mlflow_payload_omits_none_metrics() -> None:
    metadata = _metadata()
    params, metrics, tags = to_mlflow_payload(metadata)
    assert "gpu_hours" not in metrics
    assert "usd" not in metrics


def test_payload_includes_image_digest_manifest_sha256_gpu_hours_usd_instance() -> None:
    metadata = _metadata(
        image_digest="sha256:abcd1234",
        manifest_sha256="sha256:deadbeef",
        gpu_hours=4.5,
        usd=12.34,
        instance="H100-80GB",
    )
    params, metrics, tags = to_mlflow_payload(metadata)
    assert params["image_digest"] == "sha256:abcd1234"
    assert params["manifest_sha256"] == "sha256:deadbeef"
    assert metrics["gpu_hours"] == 4.5
    assert metrics["usd"] == 12.34
    assert params["instance"] == "H100-80GB"


# --- read_git_commit_sha -------------------------------------------------------------


def test_read_git_commit_sha_matches_git_rev_parse() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert read_git_commit_sha(repo_root) == expected


def test_read_git_commit_sha_raises_typed_error_when_not_a_repo(tmp_path: Path) -> None:
    with pytest.raises(TrackingBackendError):
        read_git_commit_sha(tmp_path)
