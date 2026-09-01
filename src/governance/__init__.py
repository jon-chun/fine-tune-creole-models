"""Append-only consent ledger (tech-spec §7).

tech-spec §7 names four generated governance artifacts (datasheet, model
card, consent ledger, language-readiness evidence file). This module builds
only the consent ledger — an append-only event log recording every
consent-tier grant (and later change, including withdrawal) for a source
item, keyed to item_id. No persistence backend is chosen or built here: the
ledger is always a plain list[ConsentLedgerEntry] passed in and returned,
never a file or database — a future ticket can add real storage without
this module's logic changing. The other three governance artifacts (which
need real bake-off/eval run metadata that doesn't exist yet) are out of
scope for this ticket.

Append-only means exactly that: append_entry() never mutates or removes a
prior entry, so a withdrawal after a training-tier grant is always visible
as history, not silently overwritten. current_consent_tier() is the
"what applies right now" read; ledger_for_item() is the full audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from data_contract import ConsentTier


class GovernanceError(ValueError):
    """Raised when a ConsentLedgerEntry is appended with an invalid item_id."""


@dataclass(frozen=True, slots=True)
class ConsentLedgerEntry:
    """One immutable consent-state event. Never edited or deleted after
    being appended — a change in consent is always a new entry."""

    item_id: str
    consent_tier: ConsentTier
    granted_at: datetime
    source_note: str


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


def current_consent_tier(ledger: list[ConsentLedgerEntry], item_id: str) -> ConsentTier | None:
    """The most recent consent_tier for item_id, by granted_at. On a tie,
    the entry appearing later in `ledger` wins (stable — the later append,
    not an arbitrary pick). Returns None if item_id has no entries."""
    latest: ConsentLedgerEntry | None = None
    for entry in ledger:
        if entry.item_id != item_id:
            continue
        if latest is None or entry.granted_at >= latest.granted_at:
            latest = entry
    return latest.consent_tier if latest is not None else None


def ledger_for_item(ledger: list[ConsentLedgerEntry], item_id: str) -> list[ConsentLedgerEntry]:
    """Every entry for item_id, in `ledger`'s given order. Does not sort —
    callers are expected to append in chronological order; sorting here
    would hide an out-of-order append bug rather than surface it."""
    return [entry for entry in ledger if entry.item_id == item_id]
