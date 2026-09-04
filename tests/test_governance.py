"""Tests for src/governance/ — the append-only consent ledger (tech-spec v2 §7).

Tests only external behavior: constructing ConsentLedgerEntry fixtures with
explicit datetime values, calling append_entry/current_consent/
current_training_permission/ledger_for_item, asserting on returned values.
No reliance on wall-clock time.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from governance import (
    ConsentLedgerEntry,
    GovernanceError,
    append_entry,
    current_consent,
    current_training_permission,
    ledger_for_item,
)
from governance.store import (
    LedgerStore,
    LedgerStoreError,
    WithdrawalReport,
    assert_expiry_shorter_than_sla,
    audit_run_manifest,
    delete_speaker,
    manifest_sha256,
)
from tracking import FrozenMapping, RunMetadata

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _entry(
    item_id: str,
    consent: str,
    at: datetime,
    note: str = "test",
    speaker_id: str | None = "speaker-1",
    training_permission: str = "yes_general",
) -> ConsentLedgerEntry:
    return ConsentLedgerEntry(
        item_id=item_id,
        speaker_id=speaker_id,
        consent=consent,  # type: ignore[arg-type]
        training_permission=training_permission,  # type: ignore[arg-type]
        granted_at=at,
        source_note=note,
    )


# --- append_entry: immutability and validation --------------------------------


def test_append_entry_returns_new_list_containing_the_entry() -> None:
    ledger: list[ConsentLedgerEntry] = []
    entry = _entry("item-1", "informed_consent_training", _T0)
    result = append_entry(ledger, entry)
    assert result == [entry]


def test_append_entry_does_not_mutate_original_list() -> None:
    ledger: list[ConsentLedgerEntry] = []
    entry = _entry("item-1", "informed_consent_training", _T0)
    append_entry(ledger, entry)
    assert ledger == []


def test_append_entry_preserves_existing_entries() -> None:
    first = _entry("item-1", "informed_consent_training", _T0)
    ledger = [first]
    second = _entry("item-2", "informed_consent_research", _T0 + timedelta(days=1))
    result = append_entry(ledger, second)
    assert result == [first, second]


def test_append_entry_rejects_empty_item_id() -> None:
    entry = _entry("", "informed_consent_training", _T0)
    with pytest.raises(GovernanceError):
        append_entry([], entry)


# --- ConsentLedgerEntry: required fields and validation (issue #15, MIG-01g) ----


def test_consent_ledger_entry_requires_speaker_id_and_training_permission() -> None:
    with pytest.raises(TypeError):
        ConsentLedgerEntry(  # type: ignore[call-arg]
            item_id="item-1",
            consent="informed_consent_training",
            granted_at=_T0,
            source_note="missing speaker_id and training_permission",
        )


def test_consent_ledger_entry_rejects_invalid_consent_value() -> None:
    with pytest.raises(GovernanceError):
        _entry("item-1", "vip", _T0)


def test_consent_ledger_entry_rejects_invalid_training_permission_value() -> None:
    with pytest.raises(GovernanceError):
        _entry("item-1", "informed_consent_training", _T0, training_permission="maybe")


def test_consent_ledger_entry_accepts_none_speaker_id_for_speakerless_item() -> None:
    entry = _entry("item-1", "informed_consent_training", _T0, speaker_id=None)
    assert entry.speaker_id is None


# --- current_consent -------------------------------------------------------------


def test_current_consent_returns_most_recent_by_granted_at() -> None:
    ledger = [
        _entry("item-1", "informed_consent_research", _T0),
        _entry("item-1", "informed_consent_training", _T0 + timedelta(days=1)),
    ]
    assert current_consent(ledger, "item-1") == "informed_consent_training"


def test_current_consent_returns_none_for_unknown_item() -> None:
    assert current_consent([], "item-1") is None


def test_current_consent_tie_break_is_later_appended_entry() -> None:
    ledger = [
        _entry("item-1", "informed_consent_training", _T0),
        _entry("item-1", "consent_withdrawn", _T0),
    ]
    assert current_consent(ledger, "item-1") == "consent_withdrawn"


def test_withdrawal_after_training_grant_is_reflected() -> None:
    ledger: list[ConsentLedgerEntry] = []
    ledger = append_entry(
        ledger, _entry("item-1", "informed_consent_training", _T0, "initial grant")
    )
    ledger = append_entry(
        ledger,
        _entry(
            "item-1",
            "consent_withdrawn",
            _T0 + timedelta(days=10),
            "community withdrawal request",
        ),
    )
    assert current_consent(ledger, "item-1") == "consent_withdrawn"
    # History is preserved, not overwritten.
    assert len(ledger_for_item(ledger, "item-1")) == 2


def test_current_consent_ignores_other_items() -> None:
    ledger = [
        _entry("item-1", "informed_consent_training", _T0),
        _entry("item-2", "consent_withdrawn", _T0 + timedelta(days=1)),
    ]
    assert current_consent(ledger, "item-1") == "informed_consent_training"


def test_current_consent_is_deterministic() -> None:
    ledger = [
        _entry("item-1", "informed_consent_training", _T0),
        _entry("item-1", "consent_withdrawn", _T0 + timedelta(days=1)),
    ]
    first = current_consent(ledger, "item-1")
    second = current_consent(ledger, "item-1")
    assert first == second


# --- current_training_permission (new in MIG-01g) --------------------------------


def test_current_training_permission_returns_most_recent_by_granted_at() -> None:
    ledger = [
        _entry("item-1", "informed_consent_research", _T0, training_permission="yes_scoped"),
        _entry(
            "item-1",
            "informed_consent_training",
            _T0 + timedelta(days=1),
            training_permission="yes_general",
        ),
    ]
    assert current_training_permission(ledger, "item-1") == "yes_general"


def test_current_training_permission_returns_none_for_unknown_item() -> None:
    assert current_training_permission([], "item-1") is None


def test_current_training_permission_tie_break_is_later_appended_entry() -> None:
    ledger = [
        _entry("item-1", "informed_consent_training", _T0, training_permission="yes_general"),
        _entry("item-1", "consent_withdrawn", _T0, training_permission="no"),
    ]
    assert current_training_permission(ledger, "item-1") == "no"


def test_current_training_permission_ignores_other_items() -> None:
    ledger = [
        _entry("item-1", "informed_consent_training", _T0, training_permission="yes_general"),
        _entry(
            "item-2",
            "consent_withdrawn",
            _T0 + timedelta(days=1),
            training_permission="no",
        ),
    ]
    assert current_training_permission(ledger, "item-1") == "yes_general"


# --- ledger_for_item ------------------------------------------------------------


def test_ledger_for_item_returns_only_that_items_entries_in_order() -> None:
    e1 = _entry("item-1", "informed_consent_research", _T0)
    e2 = _entry("item-2", "informed_consent_training", _T0)
    e3 = _entry("item-1", "informed_consent_training", _T0 + timedelta(days=1))
    ledger = [e1, e2, e3]
    assert ledger_for_item(ledger, "item-1") == [e1, e3]


def test_ledger_for_item_returns_empty_list_for_unknown_item() -> None:
    assert ledger_for_item([], "item-1") == []


def test_ledger_for_item_still_returns_full_entries_with_new_fields() -> None:
    entry = _entry(
        "item-1",
        "informed_consent_training",
        _T0,
        speaker_id="speaker-42",
        training_permission="yes_scoped",
    )
    ledger = [entry]
    result = ledger_for_item(ledger, "item-1")
    assert result == [entry]
    assert result[0].speaker_id == "speaker-42"
    assert result[0].training_permission == "yes_scoped"


# --- LedgerStore: JSONL persistence (backlog 0013) -------------------------------


def test_store_append_and_read_roundtrip(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    e1 = _entry("item-1", "informed_consent_training", _T0)
    e2 = _entry("item-2", "informed_consent_research", _T0 + timedelta(days=1))
    store.append(e1)
    store.append(e2)
    assert store.read_all() == [e1, e2]


def test_store_never_truncates_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = LedgerStore(path)
    store.append(_entry("item-1", "informed_consent_training", _T0))
    before = path.read_text(encoding="utf-8")

    # A second store instance over the same path must never truncate what
    # the first instance wrote (append-only at the file-handle level).
    store2 = LedgerStore(path)
    store2.append(_entry("item-2", "informed_consent_research", _T0 + timedelta(days=1)))
    after = path.read_text(encoding="utf-8")

    assert after.startswith(before)
    assert len(store2.read_all()) == 2


def test_store_rows_are_jsonl_with_iso_granted_at(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = LedgerStore(path)
    store.append(_entry("item-1", "informed_consent_training", _T0, speaker_id="speaker-1"))

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert set(row) >= {
        "item_id",
        "speaker_id",
        "consent",
        "training_permission",
        "granted_at",
        "source_note",
    }
    assert row["item_id"] == "item-1"
    assert row["speaker_id"] == "speaker-1"
    # ISO-8601 UTC string, not a datetime repr.
    assert isinstance(row["granted_at"], str)
    parsed = datetime.fromisoformat(row["granted_at"].replace("Z", "+00:00"))
    assert parsed == _T0.replace(tzinfo=timezone.utc)


def test_entries_for_speaker(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    mine1 = _entry("item-1", "informed_consent_training", _T0, speaker_id="speaker-1")
    other = _entry("item-2", "informed_consent_training", _T0, speaker_id="speaker-2")
    mine2 = _entry(
        "item-3", "informed_consent_research", _T0 + timedelta(days=1), speaker_id="speaker-1"
    )
    for entry in (mine1, other, mine2):
        store.append(entry)

    assert store.entries_for_speaker("speaker-1") == [mine1, mine2]
    assert store.entries_for_speaker("speaker-2") == [other]
    assert store.entries_for_speaker("speaker-unknown") == []
    # Store-backed counterpart of ledger_for_item.
    assert store.entries_for_item("item-1") == [mine1]


def test_verify_chain_detects_edited_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = LedgerStore(path, chained=True)
    store.append(_entry("item-1", "informed_consent_training", _T0))
    store.append(_entry("item-2", "informed_consent_research", _T0 + timedelta(days=1)))
    assert store.verify_chain() is True

    # Edit the first line in place (tamper) without going through append().
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["source_note"] = "tampered"
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert store.verify_chain() is False


# --- delete_speaker withdrawal drill (backlog 0025) -------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def test_delete_speaker_removes_items_from_every_manifest(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    store.append(_entry("item-1", "informed_consent_training", _T0, speaker_id="withdrawer"))
    store.append(_entry("item-2", "informed_consent_training", _T0, speaker_id="other"))

    manifest_a = tmp_path / "manifest_a.jsonl"
    manifest_b = tmp_path / "manifest_b.jsonl"
    _write_jsonl(
        manifest_a,
        [
            {"item_id": "item-1", "speaker_id": "withdrawer"},
            {"item_id": "item-2", "speaker_id": "other"},
        ],
    )
    _write_jsonl(manifest_b, [{"item_id": "item-1", "speaker_id": "withdrawer"}])

    report = delete_speaker(
        "withdrawer", store=store, manifest_paths=[manifest_a, manifest_b], run_manifests=[]
    )

    assert report.data_manifests_rewritten == (str(manifest_a), str(manifest_b))
    rows_a = [json.loads(ln) for ln in manifest_a.read_text(encoding="utf-8").splitlines() if ln]
    assert [row["item_id"] for row in rows_a] == ["item-2"]
    rows_b = [json.loads(ln) for ln in manifest_b.read_text(encoding="utf-8").splitlines() if ln]
    assert rows_b == []

    # A tombstone recording what was removed exists alongside each manifest.
    tombstone_a = json.loads(
        manifest_a.with_suffix(".jsonl.tombstoned").read_text(encoding="utf-8").splitlines()[0]
    )
    assert tombstone_a["speaker_id"] == "withdrawer"
    assert tombstone_a["removed_item_ids"] == ["item-1"]


def test_delete_speaker_records_withdrawal_not_history_deletion(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = LedgerStore(path)
    original = _entry(
        "item-1", "informed_consent_training", _T0, speaker_id="withdrawer", note="original grant"
    )
    store.append(original)

    report = delete_speaker("withdrawer", store=store, manifest_paths=[], run_manifests=[])

    assert report.items_withdrawn == ("item-1",)
    assert report.ledger_entries_appended == 1
    all_entries = store.entries_for_item("item-1")
    # History preserved: the original grant is still present...
    assert original in all_entries
    # ...plus a new consent_withdrawn entry, never a mutation/removal.
    assert len(all_entries) == 2
    assert current_consent(all_entries, "item-1") == "consent_withdrawn"
    assert current_training_permission(all_entries, "item-1") == "no"


def test_delete_speaker_flags_runs_whose_manifest_contained_speaker(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    store.append(_entry("item-1", "informed_consent_training", _T0, speaker_id="withdrawer"))

    run_manifest_with = tmp_path / "manifest_run-with-speaker.json"
    run_manifest_with.write_text(
        json.dumps(
            {
                "run_id": "run-with-speaker",
                "schema_version": "2.0.0",
                "items": [{"item_id": "item-1", "sha256": "aaa"}],
                "manifest_sha256": "irrelevant-for-this-test",
            }
        ),
        encoding="utf-8",
    )
    run_manifest_without = tmp_path / "manifest_run-without-speaker.json"
    run_manifest_without.write_text(
        json.dumps(
            {
                "run_id": "run-without-speaker",
                "schema_version": "2.0.0",
                "items": [{"item_id": "item-99", "sha256": "zzz"}],
                "manifest_sha256": "irrelevant-for-this-test",
            }
        ),
        encoding="utf-8",
    )

    report = delete_speaker(
        "withdrawer",
        store=store,
        manifest_paths=[],
        run_manifests=[run_manifest_with, run_manifest_without],
    )

    assert report.runs_flagged_withdrawn == ("run-with-speaker",)


def test_drill_script_dry_run_prints_report_and_changes_nothing(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    store = LedgerStore(ledger_path)
    store.append(_entry("item-1", "informed_consent_training", _T0, speaker_id="withdrawer"))

    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest_path, [{"item_id": "item-1", "speaker_id": "withdrawer"}])
    ledger_before = ledger_path.read_text(encoding="utf-8")
    manifest_before = manifest_path.read_text(encoding="utf-8")

    report = delete_speaker(
        "withdrawer", store=store, manifest_paths=[manifest_path], run_manifests=[], dry_run=True
    )

    assert isinstance(report, WithdrawalReport)
    assert report.dry_run is True
    assert report.items_withdrawn == ("item-1",)
    assert report.ledger_entries_appended == 0
    # Report names what *would* be touched, but nothing on disk changed.
    assert report.data_manifests_rewritten == (str(manifest_path),)
    assert ledger_path.read_text(encoding="utf-8") == ledger_before
    assert manifest_path.read_text(encoding="utf-8") == manifest_before
    assert not manifest_path.with_suffix(".jsonl.tombstoned").exists()


# --- manifest_sha256 / audit_run_manifest (backlog 0025) --------------------------


def test_manifest_sha256_pinned_fixture_value(tmp_path: Path) -> None:
    # Per this wave's "Shared formats fixed for this wave" section:
    # sha256 hex of json.dumps(items, sort_keys=True, separators=(",", ":"))
    # with items sorted by item_id. Fixture pinned so a change to this
    # implementation (or a sibling ticket's independent implementation, per
    # Wave 3 D8) is caught rather than silently drifting.
    manifest_path = tmp_path / "manifest_run-1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "schema_version": "2.0.0",
                # Deliberately out of item_id order; manifest_sha256 sorts.
                "items": [
                    {"item_id": "item-2", "sha256": "bbb"},
                    {"item_id": "item-1", "sha256": "aaa"},
                ],
                "manifest_sha256": "placeholder",
            }
        ),
        encoding="utf-8",
    )
    assert (
        manifest_sha256(manifest_path)
        == "c4aa698c39e753dbb2e19b4b424696abb45ad5cd4c50b67d1a56df9927c09d8d"
    )


def _run_metadata(*, manifest_sha256_value: str | None) -> RunMetadata:
    return RunMetadata(
        run_id="run-1",
        stage="train",
        language="frc",
        config=FrozenMapping({}),
        git_commit_sha="deadbeef",
        started_at=_T0,
        completed_at=_T0 + timedelta(hours=1),
        artifact_refs=(),
        seed=0,
        split_id="split-1",
        lock_hash="lockhash",
        tree_dirty=False,
        manifest_sha256=manifest_sha256_value,
    )


def test_audit_detects_tampered_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest_run-1.json"
    items = [{"item_id": "item-1", "sha256": "aaa"}]
    manifest_path.write_text(
        json.dumps(
            {"run_id": "run-1", "schema_version": "2.0.0", "items": items, "manifest_sha256": ""}
        ),
        encoding="utf-8",
    )
    correct_hash = manifest_sha256(manifest_path)
    run_metadata = _run_metadata(manifest_sha256_value=correct_hash)

    # Audit passes when the recorded hash matches what's on disk.
    assert audit_run_manifest(run_metadata, manifest_path) == correct_hash

    # Tamper with the manifest's items after the run recorded its hash.
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["items"] = [{"item_id": "item-1", "sha256": "tampered"}]
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(LedgerStoreError) as excinfo:
        audit_run_manifest(run_metadata, manifest_path)
    assert correct_hash in str(excinfo.value)
    assert "run-1" in str(excinfo.value)


# --- assert_expiry_shorter_than_sla (backlog 0013/0025) ---------------------------


def test_expiry_rule_shorter_than_30_days() -> None:
    assert_expiry_shorter_than_sla(29)
    with pytest.raises(LedgerStoreError):
        assert_expiry_shorter_than_sla(30)
    with pytest.raises(LedgerStoreError):
        assert_expiry_shorter_than_sla(31)
