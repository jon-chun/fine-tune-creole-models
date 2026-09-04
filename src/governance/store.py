"""Consent-ledger persistence (backlog 0013), `delete_speaker` withdrawal drill
and per-run manifest hash audit (backlog 0025; tech-spec v2 §7 "Withdrawal
and audit").

`src/governance/__init__.py` keeps the ledger as a plain, in-memory
`list[ConsentLedgerEntry]` (ADR 0004): callers own storage, and the module's
append-only/latest-wins logic never depends on how (or whether) a caller
persists it. This module is the "future ticket" ADR 0004 and the package
docstring both name: a real, durable backend built on top of that same
`ConsentLedgerEntry`/`append_entry` logic, without changing it.

`LedgerStore` is a JSONL file: one `ConsentLedgerEntry` per line, opened in
append mode and never truncated (ADR 0004's append-only guarantee, now
enforced at the file-handle level as well as the in-memory-list level).
`delete_speaker` never deletes ledger history either — a withdrawal is
recorded as a new `consent_withdrawn` entry, exactly like the in-memory
module's own withdrawal-after-grant example. What it *does* remove is the
speaker's rows from downstream data manifests (backlog 0013's "working
copies... and derivative packages" requirement) and it reports every
per-run coreset manifest that contained the speaker's items, so those runs
can be flagged `withdrawn` (tech-spec v2 §7 "Model release classes").

`manifest_sha256`/`audit_run_manifest` implement the "per-run manifest hash
audit" half of backlog 0025 privately: the exact hash algorithm is fixed by
this wave's "Shared formats" section (governance audits the same
`manifest_sha256` that coreset writes and tracking hashes; each ticket
implements it independently and Wave 3 D8 asserts they agree), pinned here
against a fixture in `tests/test_governance.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracking import RunMetadata

from governance import ConsentLedgerEntry, GovernanceError, append_entry

# Row keys fixed by this wave's "Shared formats fixed for this wave" section:
# exactly ConsentLedgerEntry's six fields, granted_at as an ISO-8601 UTC
# string. Order is stable (not load-bearing for parsing, since rows are
# loaded by key) but kept fixed for readable diffs.
_ROW_FIELDS = (
    "item_id",
    "speaker_id",
    "consent",
    "training_permission",
    "granted_at",
    "source_note",
)


class LedgerStoreError(GovernanceError):
    """Raised for a store-level problem: a malformed line on read, a tamper
    detected by verify_chain(), or a manifest audit mismatch."""


def _entry_to_row(entry: ConsentLedgerEntry, *, prev_sha256: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "item_id": entry.item_id,
        "speaker_id": entry.speaker_id,
        "consent": entry.consent,
        "training_permission": entry.training_permission,
        # ISO-8601 UTC string (this wave's fixed row format). A naive
        # datetime (no tzinfo) is treated as already-UTC and stamped with
        # the "Z" suffix rather than silently localized, since every caller
        # in this repo's tests constructs granted_at as naive UTC.
        "granted_at": _isoformat_utc(entry.granted_at),
        "source_note": entry.source_note,
    }
    if prev_sha256 is not None:
        row["prev_sha256"] = prev_sha256
    return row


def _isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: str) -> datetime:
    # Returned naive (tzinfo stripped), matching this repo's existing
    # convention of naive-UTC datetimes everywhere else (ConsentLedgerEntry
    # fixtures, RunMetadata.started_at/completed_at) — this store's
    # write/read roundtrip must reproduce the exact `granted_at` a caller
    # constructed with, not silently promote it to timezone-aware.
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _row_to_entry(row: dict[str, Any], *, line_number: int, path: Path) -> ConsentLedgerEntry:
    missing = [key for key in _ROW_FIELDS if key not in row]
    if missing:
        raise LedgerStoreError(f"{path}:{line_number}: missing field(s) {sorted(missing)}")
    try:
        return ConsentLedgerEntry(
            item_id=row["item_id"],
            speaker_id=row["speaker_id"],
            consent=row["consent"],
            training_permission=row["training_permission"],
            granted_at=_parse_iso_utc(row["granted_at"]),
            source_note=row["source_note"],
        )
    except GovernanceError as exc:
        raise LedgerStoreError(f"{path}:{line_number}: {exc}") from exc


def _line_sha256(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


class LedgerStore:
    """A JSONL-backed, append-only consent ledger (backlog 0013).

    Every `append()` opens the file in append mode (`"a"`), writes exactly
    one line, flushes and `os.fsync`s the file descriptor, then closes it —
    never opened in truncating (`"w"`) mode anywhere in this class, so a
    crash mid-write can at worst leave a torn final line, never lose a prior
    one. `prev_sha256` is optional hash-chaining (each row may record the
    sha256 of the previous row's raw JSON line) so `verify_chain()` can
    detect a line edited after the fact — tamper evidence, not tamper
    prevention: nothing stops a determined editor from rewriting the whole
    file and its chain, but an *isolated* edited line breaks the chain link
    to its neighbor and is caught.
    """

    def __init__(self, path: Path, *, chained: bool = False) -> None:
        self.path = Path(path)
        self.chained = chained
        if not self.path.exists():
            self.path.touch()

    def append(self, entry: ConsentLedgerEntry) -> None:
        """Validates entry.item_id via `governance.append_entry`'s own rule
        (raises GovernanceError, not just LedgerStoreError, for that case —
        the same error a caller of the in-memory API already expects), then
        appends one JSONL line."""
        append_entry([], entry)  # reuses append_entry's item_id validation; result discarded

        prev_sha256: str | None = None
        if self.chained:
            existing_lines = self._nonblank_lines()
            if existing_lines:
                prev_sha256 = _line_sha256(existing_lines[-1])

        row = _entry_to_row(entry, prev_sha256=prev_sha256)
        line = json.dumps(row, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _nonblank_lines(self) -> list[str]:
        text = self.path.read_text(encoding="utf-8")
        return [ln for ln in text.splitlines() if ln.strip()]

    def read_all(self) -> list[ConsentLedgerEntry]:
        """Every entry, in file order (append order — never sorted, matching
        `governance.ledger_for_item`'s own no-sort discipline)."""
        entries: list[ConsentLedgerEntry] = []
        for line_number, line in enumerate(self._nonblank_lines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerStoreError(f"{self.path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise LedgerStoreError(f"{self.path}:{line_number}: line is not a JSON object")
            entries.append(_row_to_entry(row, line_number=line_number, path=self.path))
        return entries

    def entries_for_speaker(self, speaker_id: str) -> list[ConsentLedgerEntry]:
        """Every entry for speaker_id, in file order."""
        return [entry for entry in self.read_all() if entry.speaker_id == speaker_id]

    def entries_for_item(self, item_id: str) -> list[ConsentLedgerEntry]:
        """Every entry for item_id, in file order (store-backed counterpart
        of `governance.ledger_for_item`)."""
        return [entry for entry in self.read_all() if entry.item_id == item_id]

    def verify_chain(self) -> bool:
        """True iff every row's recorded `prev_sha256` matches the sha256 of
        the raw line before it (chained stores only). Returns True
        unconditionally for a store opened with `chained=False` or with
        fewer than two rows — there is nothing to verify. Detects a line
        *edited in place* (its neighbor's recorded hash of it no longer
        matches); it cannot detect a whole-file rewrite that recomputes every
        chain link consistently, or the deletion of the very first row."""
        lines = self._nonblank_lines()
        if not self.chained or len(lines) < 2:
            return True
        for i in range(1, len(lines)):
            try:
                row = json.loads(lines[i])
            except json.JSONDecodeError:
                return False
            recorded_prev = row.get("prev_sha256")
            if recorded_prev != _line_sha256(lines[i - 1]):
                return False
        return True


def manifest_sha256(path: Path) -> str:
    """sha256 hex of a per-run coreset manifest JSON file's `items`, per this
    wave's "Shared formats fixed for this wave" section: sha256 of the
    UTF-8 bytes of `json.dumps(items, sort_keys=True, separators=(",",
    ":"))`, with `items` sorted by `item_id` first. Recomputed from the raw
    `items` array on disk rather than trusting the manifest's own recorded
    `manifest_sha256` field — that field is what this function is used to
    *check* (see audit_run_manifest), so it must not be an input here."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = sorted(data["items"], key=lambda item: item["item_id"])
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_run_manifest(run_metadata: RunMetadata, manifest_path: Path) -> str:
    """Recomputes manifest_sha256(manifest_path) and compares it against
    `run_metadata.manifest_sha256` (tech-spec v2 §7 "Withdrawal and audit":
    "every run records the item_id + content-hash manifest of its training
    set... hashed into the MLflow run"). Returns the matching hash on
    success. Raises LedgerStoreError naming both hashes on any mismatch,
    including when `run_metadata.manifest_sha256` is None (a run that never
    recorded a manifest hash cannot be audited as matching)."""
    recomputed = manifest_sha256(manifest_path)
    recorded = run_metadata.manifest_sha256
    if recorded != recomputed:
        raise LedgerStoreError(
            f"manifest hash mismatch for run {run_metadata.run_id}: "
            f"recorded manifest_sha256={recorded!r}, recomputed={recomputed!r} "
            f"from {manifest_path}"
        )
    return recomputed


def assert_expiry_shorter_than_sla(expiry_days: float, *, sla_days: float = 30) -> None:
    """Backlog 0013/0025's object-storage version-expiry rule: "shorter than
    the 30-day withdrawal SLA, so storage versioning cannot silently defeat
    withdrawal" (tech-spec v2 §7). Raises LedgerStoreError if
    `expiry_days >= sla_days`. The config key that supplies `expiry_days` in
    production (`configs/cloud/`) is out of scope for this ticket (backlog
    0023); this is the pure rule a config-loading ticket calls."""
    if expiry_days >= sla_days:
        raise LedgerStoreError(
            f"object-storage expiry ({expiry_days} days) must be shorter than the "
            f"withdrawal SLA ({sla_days} days)"
        )


@dataclass(frozen=True, slots=True)
class WithdrawalReport:
    """What delete_speaker() did (or, under dry_run, would do). Never
    includes deleted history — the ledger entries themselves are additions
    (a `consent_withdrawn` row per item), not removals."""

    speaker_id: str
    items_withdrawn: tuple[str, ...]
    ledger_entries_appended: int
    data_manifests_rewritten: tuple[str, ...]
    runs_flagged_withdrawn: tuple[str, ...]
    dry_run: bool


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise LedgerStoreError(f"{path}: each line must be a JSON object")
        rows.append(row)
    return rows


def delete_speaker(
    speaker_id: str,
    *,
    store: LedgerStore,
    manifest_paths: list[Path] | tuple[Path, ...] = (),
    run_manifests: list[Path] | tuple[Path, ...] = (),
    dry_run: bool = False,
) -> WithdrawalReport:
    """The withdrawal drill (backlog 0025): a speaker withdraws consent for
    every item of theirs recorded in `store`.

    1. Ledger: appends one `consent_withdrawn` entry (training_permission
       "no") per distinct item_id the speaker has any history for — history
       is never deleted, only added to (ADR 0004; matches the in-memory
       module's own "withdrawal after training grant" example). Skipped
       entirely if the speaker has no ledger entries.
    2. Data manifests (`manifest_paths`, each a `DataItem`-shaped JSONL file
       per `data_contract.read_manifest`'s input format): every row whose
       `speaker_id` matches is removed and rewritten to `<path>.tombstoned`
       recording what was removed, then the original path is rewritten
       without those rows (backlog 0013 "working copies... and derivative
       packages"). Object-storage deletion via the B2 API is out of scope
       (T-006) — these are local/working-copy JSONL files only.
    3. Run manifests (`run_manifests`, each this wave's per-run coreset
       manifest JSON): any manifest whose `items` contains one of the
       withdrawn item_ids is reported (not modified — the manifest is a
       historical record of what a completed run trained on) so its run_id
       can be flagged `withdrawn` for future releases (tech-spec v2 §7
       "Model release classes").

    `dry_run=True` computes and returns the same report without writing
    anything: no ledger append, no manifest rewrite.
    """
    withdrawn_items = sorted({entry.item_id for entry in store.entries_for_speaker(speaker_id)})

    if not dry_run:
        for item_id in withdrawn_items:
            store.append(
                ConsentLedgerEntry(
                    item_id=item_id,
                    speaker_id=speaker_id,
                    consent="consent_withdrawn",
                    training_permission="no",
                    granted_at=datetime.now(timezone.utc),
                    source_note=f"delete_speaker withdrawal drill for speaker_id={speaker_id}",
                )
            )

    rewritten_manifests: list[str] = []
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path)
        rows = _load_jsonl_rows(manifest_path)
        kept = [row for row in rows if row.get("speaker_id") != speaker_id]
        removed = [row for row in rows if row.get("speaker_id") == speaker_id]
        if not removed:
            continue
        rewritten_manifests.append(str(manifest_path))
        if dry_run:
            continue
        tombstone_path = manifest_path.with_suffix(manifest_path.suffix + ".tombstoned")
        tombstone_record = {
            "speaker_id": speaker_id,
            "removed_item_ids": sorted(row.get("item_id", "") for row in removed),
            "removed_at": _isoformat_utc(datetime.now(timezone.utc)),
            "source_manifest": str(manifest_path),
        }
        existing_tombstones = (
            _load_jsonl_rows(tombstone_path) if tombstone_path.exists() else []
        )
        with tombstone_path.open("w", encoding="utf-8") as fh:
            for record in [*existing_tombstones, tombstone_record]:
                fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        with manifest_path.open("w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    flagged_runs: list[str] = []
    withdrawn_item_set = set(withdrawn_items)
    for run_manifest_path in run_manifests:
        run_manifest_path = Path(run_manifest_path)
        data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        manifest_item_ids = {item["item_id"] for item in data.get("items", [])}
        if manifest_item_ids & withdrawn_item_set:
            run_id = data.get("run_id")
            if run_id:
                flagged_runs.append(str(run_id))

    return WithdrawalReport(
        speaker_id=speaker_id,
        items_withdrawn=tuple(withdrawn_items),
        ledger_entries_appended=0 if dry_run else len(withdrawn_items),
        data_manifests_rewritten=tuple(rewritten_manifests),
        runs_flagged_withdrawn=tuple(sorted(set(flagged_runs))),
        dry_run=dry_run,
    )
