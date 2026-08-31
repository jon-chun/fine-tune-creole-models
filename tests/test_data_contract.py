"""Tests for src/data_contract.py — the pipeline's ingestion eligibility gate.

Tests only external behavior: constructing DataItem records and calling
is_eligible() on them. Never asserts on internal representation.
"""

import socket
from unittest.mock import patch

import pytest

from data_contract import DataContractError, DataItem, is_eligible


def make_item(**overrides: object) -> DataItem:
    """A minimally valid, fully-eligible item; override fields per test."""
    defaults: dict[str, object] = dict(
        item_id="item-001",
        source="test-collection",
        language_tag="frc",
        lect=None,
        orthography_system="ad_hoc",
        consent_tier="training",
        rights="cc_open",
        training_permission="yes_general",
        cultural_sensitivity="open",
        community_review_signed_off=False,
        release_class="public",
        synthetic=False,
        generator=None,
        provenance="original",
        schema_version="1.0.0",
    )
    defaults.update(overrides)
    return DataItem(**defaults)  # type: ignore[arg-type]


# --- Construction: required fields ---------------------------------------


def test_constructing_item_with_all_required_fields_succeeds() -> None:
    item = make_item()
    assert item.item_id == "item-001"
    assert item.language_tag == "frc"


@pytest.mark.parametrize("field", ["item_id", "language_tag", "orthography_system"])
def test_missing_hard_required_field_is_rejected(field: str) -> None:
    with pytest.raises(DataContractError):
        make_item(**{field: ""})


# --- Construction: synthetic/generator invariant ---------------------------


def test_synthetic_item_without_generator_is_rejected() -> None:
    with pytest.raises(DataContractError):
        make_item(synthetic=True, generator=None)


def test_synthetic_item_with_generator_succeeds() -> None:
    item = make_item(synthetic=True, generator="rule:v1.0.0#R017")
    assert item.generator == "rule:v1.0.0#R017"


# --- Construction: closed enums reject out-of-set values -------------------


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("language_tag", "spanish"),
        ("consent_tier", "public-domain"),
        ("orthography_system", "IPA"),
        ("cultural_sensitivity", "top_secret"),
        ("release_class", "private"),
        ("rights", "trust_me"),
        ("training_permission", "maybe"),
    ],
)
def test_enum_field_rejects_value_outside_closed_set(field: str, bad_value: str) -> None:
    with pytest.raises(DataContractError):
        make_item(**{field: bad_value})


# --- is_eligible: the happy path -------------------------------------------


def test_is_eligible_true_when_every_condition_satisfied() -> None:
    result = is_eligible(make_item())
    assert result.eligible is True
    assert result.reasons == ()


# --- is_eligible: each failing condition, individually ---------------------


def test_ineligible_when_rights_not_cleared() -> None:
    result = is_eligible(make_item(rights="rights_unknown"))
    assert result.eligible is False
    assert result.reasons == ("rights_not_cleared",)


def test_ineligible_when_rights_all_reserved() -> None:
    result = is_eligible(make_item(rights="all_rights_reserved"))
    assert result.eligible is False
    assert "rights_not_cleared" in result.reasons


def test_ineligible_when_training_permission_no() -> None:
    result = is_eligible(make_item(training_permission="no"))
    assert result.eligible is False
    assert result.reasons == ("training_permission_not_granted",)


def test_ineligible_when_training_permission_uncertain() -> None:
    """Uncertain training permission is fail-safe-treated as no."""
    result = is_eligible(make_item(training_permission="uncertain"))
    assert result.eligible is False
    assert result.reasons == ("training_permission_not_granted",)


def test_ineligible_when_cultural_sensitivity_restricted() -> None:
    result = is_eligible(make_item(cultural_sensitivity="restricted"))
    assert result.eligible is False
    assert result.reasons == ("cultural_sensitivity_restricted",)


def test_ineligible_when_cultural_sensitivity_sacred() -> None:
    result = is_eligible(make_item(cultural_sensitivity="sacred"))
    assert result.eligible is False
    assert result.reasons == ("cultural_sensitivity_restricted",)


def test_ineligible_when_release_class_do_not_use() -> None:
    result = is_eligible(make_item(release_class="do_not_use"))
    assert result.eligible is False
    assert result.reasons == ("release_class_do_not_use",)


def test_ineligible_reports_multiple_failing_conditions() -> None:
    result = is_eligible(make_item(rights="rights_unknown", release_class="do_not_use"))
    assert result.eligible is False
    assert set(result.reasons) == {"rights_not_cleared", "release_class_do_not_use"}


# --- is_eligible: boundary case that must remain accepted -------------------


def test_community_review_with_signoff_is_accepted() -> None:
    """Signed-off community review must not be rejected as restricted."""
    result = is_eligible(
        make_item(cultural_sensitivity="community_review", community_review_signed_off=True)
    )
    assert result.eligible is True
    assert result.reasons == ()


def test_community_review_without_signoff_is_rejected() -> None:
    """Bare community review must not silently pass as if it were open."""
    result = is_eligible(
        make_item(cultural_sensitivity="community_review", community_review_signed_off=False)
    )
    assert result.eligible is False
    assert result.reasons == ("cultural_sensitivity_restricted",)


# --- is_eligible: purity / determinism --------------------------------------


def test_is_eligible_is_deterministic() -> None:
    item = make_item(rights="rights_unknown")
    first = is_eligible(item)
    second = is_eligible(item)
    assert first == second


def test_is_eligible_touches_no_filesystem_or_network() -> None:
    """Test-double check (Testing Decisions' named technique): patch the
    file-open and socket-connect primitives to raise if called, then run
    is_eligible across every branch. If it were to read a file or open a
    connection, this test would fail loudly instead of merely by omission."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("is_eligible must not touch the filesystem or network")

    items = [
        make_item(),
        make_item(rights="rights_unknown"),
        make_item(training_permission="uncertain"),
        make_item(cultural_sensitivity="community_review", community_review_signed_off=False),
        make_item(release_class="do_not_use"),
    ]

    with (
        patch("builtins.open", side_effect=_boom),
        patch.object(socket.socket, "connect", side_effect=_boom),
    ):
        for item in items:
            is_eligible(item)
