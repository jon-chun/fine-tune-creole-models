"""Append-only consent ledger (tech-spec v2 §7).

tech-spec v2 §7 names four generated governance artifacts (datasheet, model
card, consent ledger, language-readiness evidence file). This module builds
only the consent ledger — an append-only event log recording every
consent/training_permission grant (and later change, including withdrawal)
for a source item, keyed to both `item_id` and `speaker_id` (tech-spec v2
§7: "append-only log of every source item's `consent`/`training_permission`
grant, keyed to `item_id` and `speaker_id`"; decision brief §3 item 14). No
persistence backend is chosen or built here: the ledger is always a plain
list[ConsentLedgerEntry] passed in and returned, never a file or database —
a future ticket can add real storage without this module's logic changing
(backlog 0013). The other three governance artifacts (which need real
bake-off/eval run metadata that doesn't exist yet) are out of scope for this
ticket (backlog 0016).

Append-only means exactly that: append_entry() never mutates or removes a
prior entry, so a withdrawal after a training-permission grant is always
visible as history, not silently overwritten (ADR 0004). current_consent()
and current_training_permission() are the "what applies right now" reads;
ledger_for_item() is the full audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import get_args

from data_contract import Consent, TrainingPermission, validate_literal

_CONSENTS = frozenset(get_args(Consent))
_TRAINING_PERMISSIONS = frozenset(get_args(TrainingPermission))


class GovernanceError(ValueError):
    """Raised when a ConsentLedgerEntry is appended with an invalid item_id,
    or constructed with a consent or training_permission value outside its
    closed set (issue #15)."""


@dataclass(frozen=True, slots=True)
class ConsentLedgerEntry:
    """One immutable consent-state event. Never edited or deleted after
    being appended — a change in consent is always a new entry.

    `speaker_id` is nullable (tech-spec v2 §2.1's `DataItem.speaker_id` is
    itself nullable — a ledger entry may cover a speakerless text item) so
    this field mirrors that nullability rather than forcing a placeholder
    value (MIG-01g "Proposed resolution"). `training_permission` is a
    first-class field alongside `consent`, not merely inferred from it, so a
    future reconciliation against a `DataItem` snapshot (backlog 0017) can
    compare both independently.
    """

    item_id: str
    speaker_id: str | None
    consent: Consent
    training_permission: TrainingPermission
    granted_at: datetime
    source_note: str

    def __post_init__(self) -> None:
        # Same enum-membership discipline as DataItem's own __post_init__
        # (issue #15): a bad consent/training_permission value must be
        # rejected at construction, not silently accepted into the
        # append-only ledger.
        try:
            validate_literal(self.consent, tuple(_CONSENTS), "consent")
            validate_literal(
                self.training_permission, tuple(_TRAINING_PERMISSIONS), "training_permission"
            )
        except ValueError as exc:
            raise GovernanceError(str(exc)) from exc


def append_entry(
    ledger: list[ConsentLedgerEntry], entry: ConsentLedgerEntry
) -> list[ConsentLedgerEntry]:
    """Returns a new list with `entry` appended. Never mutates `ledger` in
    place — reinforces the append-only/immutable-history property
    structurally, not just by convention. Raises GovernanceError if
    entry.item_id is empty."""
    if not entry.item_id:
        raise GovernanceError("ConsentLedgerEntry.item_id is required and must be non-empty")
    return [*ledger, entry]


def current_consent(ledger: list[ConsentLedgerEntry], item_id: str) -> Consent | None:
    """The most recent consent value for item_id, by granted_at. On a tie,
    the entry appearing later in `ledger` wins (stable — the later append,
    not an arbitrary pick). Returns None if item_id has no entries."""
    latest: ConsentLedgerEntry | None = None
    for entry in ledger:
        if entry.item_id != item_id:
            continue
        if latest is None or entry.granted_at >= latest.granted_at:
            latest = entry
    return latest.consent if latest is not None else None


def current_training_permission(
    ledger: list[ConsentLedgerEntry], item_id: str
) -> TrainingPermission | None:
    """The most recent training_permission value for item_id, by
    granted_at. Same latest-wins tie-break as current_consent() (MIG-01g
    story 3: this must not introduce a second, subtly different resolution
    rule). Returns None if item_id has no entries."""
    latest: ConsentLedgerEntry | None = None
    for entry in ledger:
        if entry.item_id != item_id:
            continue
        if latest is None or entry.granted_at >= latest.granted_at:
            latest = entry
    return latest.training_permission if latest is not None else None


def ledger_for_item(ledger: list[ConsentLedgerEntry], item_id: str) -> list[ConsentLedgerEntry]:
    """Every entry for item_id, in `ledger`'s given order. Does not sort —
    callers are expected to append in chronological order; sorting here
    would hide an out-of-order append bug rather than surface it."""
    return [entry for entry in ledger if entry.item_id == item_id]
