"""Tests for src/governance/ — the append-only consent ledger (tech-spec §7).

Tests only external behavior: constructing ConsentLedgerEntry fixtures with
explicit datetime values, calling append_entry/current_consent_tier/
ledger_for_item, asserting on returned values. No reliance on wall-clock
time.
"""

from datetime import datetime, timedelta

import pytest

from governance import (
    ConsentLedgerEntry,
    GovernanceError,
    append_entry,
    current_consent_tier,
    ledger_for_item,
)

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _entry(item_id: str, tier: str, at: datetime, note: str = "test") -> ConsentLedgerEntry:
    return ConsentLedgerEntry(item_id=item_id, consent_tier=tier, granted_at=at, source_note=note)  # type: ignore[arg-type]


# --- append_entry: immutability and validation --------------------------------


def test_append_entry_returns_new_list_containing_the_entry() -> None:
    ledger: list[ConsentLedgerEntry] = []
    entry = _entry("item-1", "training", _T0)
    result = append_entry(ledger, entry)
    assert result == [entry]


def test_append_entry_does_not_mutate_original_list() -> None:
    ledger: list[ConsentLedgerEntry] = []
    entry = _entry("item-1", "training", _T0)
    append_entry(ledger, entry)
    assert ledger == []


def test_append_entry_preserves_existing_entries() -> None:
    first = _entry("item-1", "training", _T0)
    ledger = [first]
    second = _entry("item-2", "research", _T0 + timedelta(days=1))
    result = append_entry(ledger, second)
    assert result == [first, second]


def test_append_entry_rejects_empty_item_id() -> None:
    entry = _entry("", "training", _T0)
    with pytest.raises(GovernanceError):
        append_entry([], entry)


# --- current_consent_tier ------------------------------------------------------


def test_current_consent_tier_returns_most_recent_by_granted_at() -> None:
    ledger = [
        _entry("item-1", "research", _T0),
        _entry("item-1", "training", _T0 + timedelta(days=1)),
    ]
    assert current_consent_tier(ledger, "item-1") == "training"


def test_current_consent_tier_returns_none_for_unknown_item() -> None:
    assert current_consent_tier([], "item-1") is None


def test_current_consent_tier_tie_break_is_later_appended_entry() -> None:
    ledger = [
        _entry("item-1", "training", _T0),
        _entry("item-1", "withdrawal", _T0),
    ]
    assert current_consent_tier(ledger, "item-1") == "withdrawal"


def test_withdrawal_after_training_grant_is_reflected() -> None:
    ledger: list[ConsentLedgerEntry] = []
    ledger = append_entry(ledger, _entry("item-1", "training", _T0, "initial grant"))
    ledger = append_entry(
        ledger, _entry("item-1", "withdrawal", _T0 + timedelta(days=10), "community withdrawal request")
    )
    assert current_consent_tier(ledger, "item-1") == "withdrawal"
    # History is preserved, not overwritten.
    assert len(ledger_for_item(ledger, "item-1")) == 2


def test_current_consent_tier_ignores_other_items() -> None:
    ledger = [
        _entry("item-1", "training", _T0),
        _entry("item-2", "withdrawal", _T0 + timedelta(days=1)),
    ]
    assert current_consent_tier(ledger, "item-1") == "training"


# --- ledger_for_item ------------------------------------------------------------


def test_ledger_for_item_returns_only_that_items_entries_in_order() -> None:
    e1 = _entry("item-1", "research", _T0)
    e2 = _entry("item-2", "training", _T0)
    e3 = _entry("item-1", "training", _T0 + timedelta(days=1))
    ledger = [e1, e2, e3]
    assert ledger_for_item(ledger, "item-1") == [e1, e3]


def test_ledger_for_item_returns_empty_list_for_unknown_item() -> None:
    assert ledger_for_item([], "item-1") == []


# --- determinism -----------------------------------------------------------------


def test_current_consent_tier_is_deterministic() -> None:
    ledger = [_entry("item-1", "training", _T0), _entry("item-1", "withdrawal", _T0 + timedelta(days=1))]
    first = current_consent_tier(ledger, "item-1")
    second = current_consent_tier(ledger, "item-1")
    assert first == second
