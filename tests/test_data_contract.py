"""Tests for src/data_contract.py — the pipeline's ingestion eligibility gate
(contract v2 / annotation schema v2, MIG-01a / issue #25).

Tests only external behavior: constructing DataItem records, calling
is_eligible() on them, and the derive_release_class/derive_cloud_ok pure
functions. Never asserts on internal representation.
"""

import json
import socket
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from data_contract import (
    DataContractError,
    DataItem,
    ReleaseClassInputs,
    derive_cloud_ok,
    derive_release_class,
    is_eligible,
    read_manifest,
)


def make_item(**overrides: object) -> DataItem:
    """A minimally valid, fully-eligible v2 item; override fields per test.

    Unless the caller overrides `release_class` and/or `cloud_ok` explicitly,
    both are (re)derived from the other (possibly overridden) fields via the
    real derive_release_class/derive_cloud_ok functions, so a test overriding
    an upstream field (rights/consent/etc.) doesn't also need to hand-compute
    the two downstream computed fields. A test that wants to exercise the
    disagreement check passes release_class/cloud_ok explicitly alongside
    upstream fields that disagree with them.
    """
    defaults: dict[str, object] = dict(
        item_id="item-001",
        source="test-collection",
        record_type="text",
        language_tag="frc",
        eng_dialect=None,
        lect=None,
        orthography_system="ad_hoc",
        genre="other",
        register="unknown",
        rights="cc_open",
        consent="informed_consent_training",
        training_permission="yes_general",
        cultural_sensitivity="open",
        community_review_signed_off=False,
        sensitivity_tier="S0",
        access_tier=1,
        object_tier="T0",
        speaker_id=None,
        speaker_generation="unknown",
        speaker_role="other",
        gender="other_unknown",
        attribution_mode="anonymous",
        pii_status="none",
        reading_type=None,
        passage_id=None,
        pair_id=None,
        split="gold_train",
        data_class="gold",
        synthetic=False,
        generator=None,
        provenance="original",
        normalizer_status="not_ready",
        normalization_difficulty="low",
        diff_catalog_flags=[],
        schema_version="2.0.0",
    )
    defaults.update(overrides)

    if "release_class" not in overrides:
        defaults["release_class"] = derive_release_class(
            ReleaseClassInputs(
                rights=defaults["rights"],  # type: ignore[arg-type]
                training_permission=defaults["training_permission"],  # type: ignore[arg-type]
                consent=defaults["consent"],  # type: ignore[arg-type]
                cultural_sensitivity=defaults["cultural_sensitivity"],  # type: ignore[arg-type]
                community_review_signed_off=defaults["community_review_signed_off"],  # type: ignore[arg-type]
            )
        )
    if "cloud_ok" not in overrides:
        defaults["cloud_ok"] = derive_cloud_ok(
            release_class=defaults["release_class"],  # type: ignore[arg-type]
            training_permission=defaults["training_permission"],  # type: ignore[arg-type]
            sensitivity_tier=defaults["sensitivity_tier"],  # type: ignore[arg-type]
            pii_status=defaults["pii_status"],  # type: ignore[arg-type]
        )

    return DataItem(**defaults)  # type: ignore[arg-type]


# --- Construction: required fields ---------------------------------------


def test_constructing_item_with_all_required_fields_succeeds() -> None:
    item = make_item()
    assert item.item_id == "item-001"
    assert item.language_tag == "frc"


@pytest.mark.parametrize("field", ["item_id", "source", "language_tag", "orthography_system", "provenance"])
def test_missing_hard_required_field_is_rejected(field: str) -> None:
    with pytest.raises(DataContractError):
        make_item(**{field: ""})


# --- Construction: synthetic/generator invariant ---------------------------


def test_synthetic_item_without_generator_is_rejected() -> None:
    with pytest.raises(DataContractError):
        make_item(synthetic=True, generator=None, data_class="synthetic")


def test_synthetic_item_with_generator_succeeds() -> None:
    item = make_item(synthetic=True, generator="rule:v1.0.0#R017", data_class="synthetic")
    assert item.generator == "rule:v1.0.0#R017"


# --- Construction: data_class/synthetic invariant (issue #11, carried to v2) -


def test_synthetic_true_requires_data_class_synthetic() -> None:
    with pytest.raises(DataContractError):
        make_item(synthetic=True, generator="rule:v1.0.0#R017", data_class="gold")


def test_data_class_synthetic_requires_synthetic_true() -> None:
    with pytest.raises(DataContractError):
        make_item(synthetic=False, data_class="synthetic")


def test_data_class_gold_with_synthetic_false_succeeds() -> None:
    item = make_item(synthetic=False, data_class="gold")
    assert item.data_class == "gold"


def test_bronze_data_class_accepted() -> None:
    item = make_item(synthetic=False, data_class="bronze")
    assert item.data_class == "bronze"


def test_data_class_field_replaces_tier_no_alias() -> None:
    """The v1 `tier` kwarg is not a compatibility alias for `data_class` —
    passing it must raise TypeError (issue #25 story 6: hard rename, no
    shim)."""
    with pytest.raises(TypeError):
        make_item(tier="gold")


# --- Construction: closed enums reject out-of-set values -------------------


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("language_tag", "spanish"),
        ("consent", "public-domain"),
        ("orthography_system", "IPA"),
        ("cultural_sensitivity", "top_secret"),
        ("rights", "trust_me"),
        ("training_permission", "maybe"),
        ("data_class", "unreviewed"),
        ("normalizer_status", "maybe_ready"),
        ("record_type", "video"),
        ("genre", "gossip"),
        ("register", "loud"),
        ("sensitivity_tier", "S9"),
        ("access_tier", 9),
        ("object_tier", "T9"),
        ("speaker_generation", "senior_fluent"),
        ("speaker_role", "translator"),
        ("gender", "unspecified"),
        ("attribution_mode", "secret"),
        ("pii_status", "maybe"),
        ("split", "test"),
        ("normalization_difficulty", "extreme"),
    ],
)
def test_enum_field_rejects_value_outside_closed_set(field: str, bad_value: object) -> None:
    with pytest.raises(DataContractError):
        make_item(**{field: bad_value})


def test_cultural_sensitivity_drops_sacred() -> None:
    """v2 drops the v1 `sacred` value from CulturalSensitivity (ADR 0009)."""
    with pytest.raises(DataContractError):
        make_item(cultural_sensitivity="sacred")


def test_schema_version_must_be_2_0_0() -> None:
    with pytest.raises(DataContractError):
        make_item(schema_version="1.9.9")


# --- Construction: release_class / cloud_ok are computed, not free-set -----


def test_release_class_disagreeing_with_derivation_is_rejected() -> None:
    with pytest.raises(DataContractError, match="release_class"):
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            release_class="do_not_use",
        )


def test_cloud_ok_disagreeing_with_derivation_is_rejected() -> None:
    with pytest.raises(DataContractError, match="cloud_ok"):
        make_item(
            release_class="public_train_ok",
            training_permission="yes_general",
            sensitivity_tier="S0",
            pii_status="none",
            cloud_ok=False,
        )


# --- Construction: nullable fields may be omitted/None ----------------------


@pytest.mark.parametrize(
    "field", ["eng_dialect", "lect", "speaker_id", "reading_type", "passage_id", "pair_id", "generator"]
)
def test_nullable_field_accepts_none(field: str) -> None:
    item = make_item(**{field: None})
    assert getattr(item, field) is None


def test_eng_dialect_meaningful_only_with_eng_language_tag() -> None:
    item = make_item(
        language_tag="eng",
        eng_dialect="aae",
        rights="cc_open",
        training_permission="yes_general",
        consent="informed_consent_training",
        cultural_sensitivity="open",
        release_class="public_train_ok",
    )
    assert item.eng_dialect == "aae"


# --- Alias normalization (module-level _ALIASES; applied via read_manifest) -


def _preprocess_row(**overrides: object) -> dict[str, object]:
    """Mimics utils/fine_tune_cajun_preprocess.py's eligible_rows entry:
    asdict(item) plus the code_switch_spans key that module adds for its own
    downstream stages."""
    item = make_item()
    row: dict[str, object] = asdict(item)
    row["code_switch_spans"] = [{"start": 0, "end": 3, "language_tag": "eng"}]
    row.update(overrides)
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    manifest_path = path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return manifest_path


def test_alias_lf_normalizes_to_frc(tmp_path: Path) -> None:
    row = _preprocess_row(language_tag="lf")
    manifest_path = _write_manifest(tmp_path, [row])
    items = read_manifest(manifest_path)
    assert items[0].language_tag == "frc"


def test_alias_elder_l1_normalizes_to_elder_fluent(tmp_path: Path) -> None:
    row = _preprocess_row(speaker_generation="elder_L1")
    manifest_path = _write_manifest(tmp_path, [row])
    items = read_manifest(manifest_path)
    assert items[0].speaker_generation == "elder_fluent"


def test_alias_new_speaker_normalizes_to_learner_revitalization(tmp_path: Path) -> None:
    row = _preprocess_row(speaker_generation="new_speaker")
    manifest_path = _write_manifest(tmp_path, [row])
    items = read_manifest(manifest_path)
    assert items[0].speaker_generation == "learner_revitalization"


def test_alias_dlf_normalized_normalizes(tmp_path: Path) -> None:
    row = _preprocess_row(orthography_system="DLF-normalized")
    manifest_path = _write_manifest(tmp_path, [row])
    items = read_manifest(manifest_path)
    assert items[0].orthography_system == "dlf_normalized"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("KVO", "kvo"),
        ("French-like", "french_like"),
        ("English-phonetic", "english_phonetic"),
    ],
)
def test_alias_v1_orthography_spellings_normalize(tmp_path: Path, raw: str, expected: str) -> None:
    row = _preprocess_row(orthography_system=raw)
    manifest_path = _write_manifest(tmp_path, [row])
    items = read_manifest(manifest_path)
    assert items[0].orthography_system == expected


# --- Alias normalization on direct DataItem(...) construction --------------
#
# _ALIASES must apply on every construction path (issue #25 "Proposed
# resolution"), not just read_manifest — these mirror the read_manifest alias
# tests above but call make_item()/DataItem(...) directly, with no file I/O.


def test_alias_lf_normalizes_to_frc_on_direct_construction() -> None:
    item = make_item(language_tag="lf")
    assert item.language_tag == "frc"


def test_alias_v1_orthography_spellings_normalize_on_direct_construction() -> None:
    for raw, expected in (
        ("KVO", "kvo"),
        ("French-like", "french_like"),
        ("English-phonetic", "english_phonetic"),
        ("DLF-normalized", "dlf_normalized"),
    ):
        item = make_item(orthography_system=raw)
        assert item.orthography_system == expected


def test_alias_speaker_generation_normalizes_on_direct_construction() -> None:
    elder = make_item(speaker_generation="elder_L1")
    assert elder.speaker_generation == "elder_fluent"

    new_speaker = make_item(speaker_generation="new_speaker")
    assert new_speaker.speaker_generation == "learner_revitalization"


# --- derive_release_class: one test per branch of contract v2 §1 -----------


def test_derive_release_class_public_train_ok() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="cc_open",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            community_review_signed_off=False,
        )
    )
    assert result == "public_train_ok"


def test_derive_release_class_public_eval_only_via_scoped_permission() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="cc_open",
            training_permission="yes_scoped",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            community_review_signed_off=False,
        )
    )
    assert result == "public_eval_only"


def test_derive_release_class_public_eval_only_via_cleared_cc_restricted() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="cc_restricted",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            community_review_signed_off=False,
        )
    )
    assert result == "public_eval_only"


def test_derive_release_class_internal_eval_only_via_archive_permission() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="archive_permission",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            community_review_signed_off=False,
        )
    )
    assert result == "internal_eval_only"


def test_derive_release_class_internal_eval_only_via_informed_consent_research() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="cc_open",
            training_permission="yes_general",
            consent="informed_consent_research",
            cultural_sensitivity="open",
            community_review_signed_off=False,
        )
    )
    assert result == "internal_eval_only"


def test_derive_release_class_do_not_use_on_rights_unknown() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="rights_unknown",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            community_review_signed_off=False,
        )
    )
    assert result == "do_not_use"


def test_derive_release_class_do_not_use_on_consent_withdrawn() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="cc_open",
            training_permission="yes_general",
            consent="consent_withdrawn",
            cultural_sensitivity="open",
            community_review_signed_off=False,
        )
    )
    assert result == "do_not_use"


def test_derive_release_class_do_not_use_on_cultural_sensitivity_restricted() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="cc_open",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="restricted",
            community_review_signed_off=False,
        )
    )
    assert result == "do_not_use"


def test_derive_release_class_do_not_use_on_unreviewed_legacy_no_consent() -> None:
    result = derive_release_class(
        ReleaseClassInputs(
            rights="cc_open",
            training_permission="yes_general",
            consent="legacy_no_consent",
            cultural_sensitivity="open",
            community_review_signed_off=False,
        )
    )
    assert result == "do_not_use"


# --- derive_cloud_ok: one test per clause of tech-spec v2 §2.3 -------------


def test_derive_cloud_ok_true_when_public_train_ok_and_s0() -> None:
    assert (
        derive_cloud_ok(
            release_class="public_train_ok",
            training_permission="yes_general",
            sensitivity_tier="S0",
            pii_status="none",
        )
        is True
    )


def test_derive_cloud_ok_false_when_sensitivity_tier_s2() -> None:
    assert (
        derive_cloud_ok(
            release_class="public_train_ok",
            training_permission="yes_general",
            sensitivity_tier="S2",
            pii_status="none",
        )
        is False
    )


def test_derive_cloud_ok_s1_requires_pii_not_tagged() -> None:
    assert (
        derive_cloud_ok(
            release_class="public_train_ok",
            training_permission="yes_general",
            sensitivity_tier="S1",
            pii_status="tagged",
        )
        is False
    )
    assert (
        derive_cloud_ok(
            release_class="public_train_ok",
            training_permission="yes_general",
            sensitivity_tier="S1",
            pii_status="redacted",
        )
        is True
    )


def test_derive_cloud_ok_false_when_training_permission_yes_scoped() -> None:
    assert (
        derive_cloud_ok(
            release_class="public_eval_only",
            training_permission="yes_scoped",
            sensitivity_tier="S0",
            pii_status="none",
        )
        is False
    )


def test_derive_cloud_ok_false_when_release_class_not_public() -> None:
    assert (
        derive_cloud_ok(
            release_class="internal_eval_only",
            training_permission="yes_general",
            sensitivity_tier="S0",
            pii_status="none",
        )
        is False
    )


# --- is_eligible: the happy path -------------------------------------------


def test_is_eligible_true_when_every_condition_satisfied() -> None:
    result = is_eligible(make_item())
    assert result.eligible is True
    assert result.reasons == ()


# --- is_eligible: each failing condition, individually ---------------------


def test_ineligible_when_rights_not_cleared() -> None:
    result = is_eligible(
        make_item(
            rights="rights_unknown",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            release_class="do_not_use",
        )
    )
    assert result.eligible is False
    assert "rights_not_cleared" in result.reasons


def test_ineligible_when_rights_all_reserved() -> None:
    result = is_eligible(
        make_item(
            rights="all_rights_reserved",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            release_class="do_not_use",
        )
    )
    assert result.eligible is False
    assert "rights_not_cleared" in result.reasons


def test_ineligible_when_training_permission_no() -> None:
    result = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="no",
            consent="informed_consent_training",
            cultural_sensitivity="open",
        )
    )
    assert result.eligible is False
    assert "training_permission_not_granted" in result.reasons


def test_ineligible_when_training_permission_uncertain() -> None:
    """uncertain is fail-safe-treated as no, per the linguist-handoff source."""
    result = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="uncertain",
            consent="informed_consent_training",
            cultural_sensitivity="open",
        )
    )
    assert result.eligible is False
    assert "training_permission_not_granted" in result.reasons


def test_ineligible_when_cultural_sensitivity_restricted() -> None:
    result = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="restricted",
            release_class="do_not_use",
        )
    )
    assert result.eligible is False
    assert "cultural_sensitivity_restricted" in result.reasons


def test_ineligible_when_release_class_do_not_use() -> None:
    result = is_eligible(
        make_item(
            rights="rights_unknown",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            release_class="do_not_use",
        )
    )
    assert result.eligible is False
    assert "release_class_do_not_use" in result.reasons


# --- is_eligible: consent gate (tech-spec v2 §2.2 consent clause) ----------


def test_is_eligible_consent_not_granted_reason_on_consent_pending() -> None:
    result = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="consent_pending",
            cultural_sensitivity="open",
            release_class="do_not_use",
        )
    )
    assert result.eligible is False
    assert result.reasons == ("release_class_do_not_use", "consent_not_granted")


def test_ineligible_when_consent_withdrawn() -> None:
    result = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="consent_withdrawn",
            cultural_sensitivity="open",
            release_class="do_not_use",
        )
    )
    assert result.eligible is False
    assert "consent_not_granted" in result.reasons


def test_is_eligible_legacy_no_consent_requires_community_review_signed_off() -> None:
    not_signed_off = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="legacy_no_consent",
            cultural_sensitivity="open",
            community_review_signed_off=False,
            release_class="do_not_use",
        )
    )
    assert not_signed_off.eligible is False
    assert "consent_not_granted" in not_signed_off.reasons

    signed_off = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="legacy_no_consent",
            cultural_sensitivity="open",
            community_review_signed_off=True,
        )
    )
    assert signed_off.eligible is True
    assert signed_off.reasons == ()


def test_is_eligible_informed_consent_research_is_eligible() -> None:
    """informed_consent_research stays training-eligible for this
    non-commercial research project (the owner's widening), but always
    derives release_class = internal_eval_only, so it is never cloud_ok."""
    item = make_item(
        rights="cc_open",
        training_permission="yes_general",
        consent="informed_consent_research",
        cultural_sensitivity="open",
        release_class="internal_eval_only",
        cloud_ok=False,
    )
    result = is_eligible(item)
    assert result.eligible is True
    assert result.reasons == ()
    assert item.cloud_ok is False


def test_ineligible_reports_multiple_failing_conditions() -> None:
    result = is_eligible(
        make_item(
            rights="rights_unknown",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            release_class="do_not_use",
        )
    )
    assert result.eligible is False
    assert set(result.reasons) == {"rights_not_cleared", "release_class_do_not_use"}


# --- is_eligible: boundary case that must remain accepted -------------------


def test_community_review_with_signoff_is_accepted() -> None:
    """community_review-with-signoff (tech-spec v2 §2.2's exact eligible
    condition) must not be over-strictly rejected alongside restricted/
    consent_pending."""
    result = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="community_review",
            community_review_signed_off=True,
        )
    )
    assert result.eligible is True
    assert result.reasons == ()


def test_community_review_without_signoff_is_rejected() -> None:
    """The eligible condition is community_review-WITH-SIGNOFF specifically —
    bare community_review, not yet signed off, must not silently pass as if
    it were `open`."""
    result = is_eligible(
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="community_review",
            community_review_signed_off=False,
        )
    )
    assert result.eligible is False
    assert "cultural_sensitivity_restricted" in result.reasons


# --- is_eligible: purity / determinism --------------------------------------


def test_is_eligible_is_deterministic() -> None:
    item = make_item(
        rights="rights_unknown",
        training_permission="yes_general",
        consent="informed_consent_training",
        cultural_sensitivity="open",
        release_class="do_not_use",
    )
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
        make_item(
            rights="rights_unknown",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            release_class="do_not_use",
        ),
        make_item(
            rights="cc_open",
            training_permission="uncertain",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            release_class="do_not_use",
        ),
        make_item(
            rights="cc_open",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="community_review",
            community_review_signed_off=False,
            release_class="do_not_use",
        ),
        make_item(
            rights="rights_unknown",
            training_permission="yes_general",
            consent="informed_consent_training",
            cultural_sensitivity="open",
            release_class="do_not_use",
        ),
    ]

    with (
        patch("builtins.open", side_effect=_boom),
        patch.object(socket.socket, "connect", side_effect=_boom),
    ):
        for item in items:
            is_eligible(item)


# --- read_manifest: v1 rejection (issue #25 schema_version gate) -----------


def test_v1_record_rejected_with_migration_error(tmp_path: Path) -> None:
    row = _preprocess_row(schema_version="1.1.0")
    manifest_path = _write_manifest(tmp_path, [row])
    with pytest.raises(DataContractError, match=r"schema_version='1\.1\.0'.*migrate via MIG-01"):
        read_manifest(manifest_path)


# --- read_manifest: round-tripping real preprocess output (issue #18) -------


def test_read_manifest_v2_shape_round_trips(tmp_path: Path) -> None:
    """A fixture manifest row built from the full 36-field v2 shape (via
    dataclasses.asdict) round-trips through read_manifest."""
    manifest_path = _write_manifest(tmp_path, [_preprocess_row(item_id="row-1")])
    items = read_manifest(manifest_path)
    assert len(items) == 1
    assert items[0].item_id == "row-1"
    assert isinstance(items[0], DataItem)


def test_read_manifest_reads_real_preprocess_output_shape(tmp_path: Path) -> None:
    """The exact failure R10 named: DataItem(**row) raises TypeError on a real
    preprocess row because of the extra code_switch_spans key. read_manifest
    must handle that shape directly."""
    manifest_path = _write_manifest(tmp_path, [_preprocess_row(item_id="row-1")])
    items = read_manifest(manifest_path)
    assert len(items) == 1
    assert items[0].item_id == "row-1"
    assert isinstance(items[0], DataItem)


def test_read_manifest_drops_unknown_keys_by_default(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_preprocess_row(item_id="row-1")])
    items = read_manifest(manifest_path)
    # No DataItem field is named code_switch_spans; a successful construction
    # already proves it was dropped, not passed through as an unexpected
    # constructor kwarg.
    assert items[0].item_id == "row-1"


def test_read_manifest_strict_true_rejects_unknown_keys(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_preprocess_row(item_id="row-1")])
    with pytest.raises(DataContractError, match="code_switch_spans"):
        read_manifest(manifest_path, strict=True)


def test_read_manifest_multiple_rows(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path, [_preprocess_row(item_id="row-1"), _preprocess_row(item_id="row-2")]
    )
    items = read_manifest(manifest_path)
    assert [item.item_id for item in items] == ["row-1", "row-2"]


def test_read_manifest_missing_required_field_raises_with_line_and_key(tmp_path: Path) -> None:
    row = _preprocess_row(item_id="row-1")
    del row["rights"]
    manifest_path = _write_manifest(tmp_path, [row])
    with pytest.raises(DataContractError, match=r"manifest\.jsonl:1.*rights"):
        read_manifest(manifest_path)


def test_read_manifest_missing_nullable_field_does_not_raise(tmp_path: Path) -> None:
    row = _preprocess_row(item_id="row-1")
    del row["eng_dialect"]
    manifest_path = _write_manifest(tmp_path, [row])
    items = read_manifest(manifest_path)
    assert items[0].eng_dialect is None


def test_read_manifest_invalid_field_value_raises_data_contract_error(tmp_path: Path) -> None:
    row = _preprocess_row(item_id="row-1", rights="trust_me")
    manifest_path = _write_manifest(tmp_path, [row])
    with pytest.raises(DataContractError):
        read_manifest(manifest_path)


def test_read_manifest_invalid_json_line_raises_with_line_number(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DataContractError, match=r"manifest\.jsonl:1"):
        read_manifest(manifest_path)


def test_read_manifest_skips_blank_lines(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    row = _preprocess_row(item_id="row-1")
    manifest_path.write_text(f"\n{json.dumps(row)}\n\n", encoding="utf-8")
    items = read_manifest(manifest_path)
    assert len(items) == 1


def test_read_manifest_empty_file_returns_empty_list(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")
    assert read_manifest(manifest_path) == []
