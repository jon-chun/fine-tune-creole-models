"""Tests for src/coreset/ — the coverage scorecard (tech-spec v2 §8).

Tests only external behavior: constructing DataItem/DiffCatalogCell/
CoverageTargets fixtures, calling compute_coverage_scorecard(), asserting on
the returned CoverageScorecard. load_diff_catalog_cells() is tested
separately against the real committed diff-catalog YAML.

v2 migration (MIG-01f, issue #30): DataItem fixtures are rebuilt on the v2
36-field shape (data_contract.DataItem, contract v2) via
derive_release_class/derive_cloud_ok rather than the old v1 hand-set
`tier`/`consent_tier`/`release_class` kwargs.
"""

import json
from pathlib import Path

import pytest

from data_contract import (
    DataContractError,
    DataItem,
    ReleaseClassInputs,
    derive_cloud_ok,
    derive_release_class,
    read_manifest,
)
from coreset import (
    CoresetConfigError,
    CoverageTargets,
    DiffCatalogCell,
    compute_coverage_scorecard,
    load_diff_catalog_cells,
    scorecard_from_dict,
    scorecard_from_json,
    scorecard_to_dict,
    scorecard_to_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DIFF_CATALOG_DIR = REPO_ROOT / "configs" / "diff_catalog"
REAL_FRC_CATALOG = DIFF_CATALOG_DIR / "frc_vs_fra.yml"
REAL_LOU_VS_FRA_CATALOG = DIFF_CATALOG_DIR / "lou_vs_fra.yml"
REAL_LOU_VS_HAT_CATALOG = DIFF_CATALOG_DIR / "lou_vs_hat.yml"
REAL_FRC_VS_FRA_PHON_CATALOG = DIFF_CATALOG_DIR / "frc_vs_fra_phon.yml"


def _item(
    item_id: str,
    language_tag: str = "frc",
    eligible: bool = True,
    speaker_id: str | None = "spk-lafourche-1",
    lect: str | None = "Lafourche",
    genre: str = "conversation",
    speaker_generation: str = "elder_fluent",
) -> DataItem:
    """A real v2 DataItem, built the same way every other module's fixtures
    do post-MIG-01a: rights/consent/training_permission/cultural_sensitivity
    chosen so `eligible` controls both is_eligible() and, through
    derive_release_class, the required release_class field — never hand-set
    to a disagreeing value (DataItem.__post_init__ would raise)."""
    rights = "cc_open" if eligible else "rights_unknown"
    training_permission = "yes_general" if eligible else "no"
    consent = "informed_consent_training"
    cultural_sensitivity = "open"
    community_review_signed_off = False
    release_class = derive_release_class(
        ReleaseClassInputs(
            rights=rights,  # type: ignore[arg-type]
            training_permission=training_permission,  # type: ignore[arg-type]
            consent=consent,  # type: ignore[arg-type]
            cultural_sensitivity=cultural_sensitivity,  # type: ignore[arg-type]
            community_review_signed_off=community_review_signed_off,
        )
    )
    sensitivity_tier = "S0"
    pii_status = "none"
    cloud_ok = derive_cloud_ok(
        release_class=release_class,
        training_permission=training_permission,  # type: ignore[arg-type]
        sensitivity_tier=sensitivity_tier,  # type: ignore[arg-type]
        pii_status=pii_status,  # type: ignore[arg-type]
    )
    return DataItem(
        item_id=item_id,
        source="test-collection",
        record_type="text",
        language_tag=language_tag,  # type: ignore[arg-type]
        eng_dialect=None,
        lect=lect if eligible else None,
        orthography_system="ad_hoc",
        genre=genre,  # type: ignore[arg-type]
        register="casual",
        rights=rights,  # type: ignore[arg-type]
        consent=consent,  # type: ignore[arg-type]
        training_permission=training_permission,  # type: ignore[arg-type]
        cultural_sensitivity=cultural_sensitivity,  # type: ignore[arg-type]
        community_review_signed_off=community_review_signed_off,
        sensitivity_tier=sensitivity_tier,  # type: ignore[arg-type]
        access_tier=1,
        object_tier="T0",
        release_class=release_class,
        speaker_id=speaker_id if eligible else None,
        speaker_generation=speaker_generation,  # type: ignore[arg-type]
        speaker_role="interviewee",
        gender="other_unknown",
        attribution_mode="anonymous",
        pii_status=pii_status,  # type: ignore[arg-type]
        reading_type=None,
        passage_id=None,
        pair_id=None,
        split="silver_unreviewed",
        data_class="gold",
        synthetic=False,
        generator=None,
        provenance="original",
        normalizer_status="not_ready",
        normalization_difficulty="low",
        diff_catalog_flags=[],
        cloud_ok=cloud_ok,
        schema_version="2.0.0",
    )


def _cell(
    id_: str,
    priority: str,
    coverage_status: str,
    *,
    gate_class: int | None = None,
    base_failure_rate: float | None = None,
) -> DiffCatalogCell:
    return DiffCatalogCell(
        id=id_,
        axis="morphosyntax",
        feature="test feature",
        priority=priority,  # type: ignore[arg-type]
        coverage_status=coverage_status,  # type: ignore[arg-type]
        gate_class=gate_class,
        base_failure_rate=base_failure_rate,
    )


def _targets() -> CoverageTargets:
    return CoverageTargets(
        floor={"items": 5, "speakers": 3},
        aspirational={"items": 20, "speakers": 5},
    )


# --- eligibility/language filtering -------------------------------------------


def test_eligible_item_of_target_language_is_counted() -> None:
    items = [_item("a", language_tag="frc", eligible=True)]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.language == "frc"


def test_ineligible_item_is_not_counted() -> None:
    items = [_item("a", language_tag="frc", eligible=False)]
    cells = [_cell("X-001", "high", "unmet")]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), cells,
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    # Coverage is diff-catalog-status-driven, not item-count-driven; the
    # eligibility filter's effect is verified via item-count-derived fields
    # once those exist downstream — here we assert it doesn't error and
    # doesn't silently include the ineligible item as eligible-language data.
    assert scorecard.diff_catalog_coverage == {"X-001": "unmet"}


def test_item_of_different_language_is_not_counted() -> None:
    items = [_item("a", language_tag="lou", eligible=True)]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.language == "frc"


# --- observed_counts -----------------------------------------------------------


def test_observed_counts_counts_distinct_eligible_item_ids() -> None:
    items = [_item("a"), _item("b"), _item("c")]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["items"] == 3


def test_observed_counts_excludes_ineligible_items() -> None:
    items = [_item("a", eligible=True), _item("b", eligible=False)]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["items"] == 1


def test_observed_counts_excludes_other_language_items() -> None:
    items = [_item("a", language_tag="frc"), _item("b", language_tag="lou")]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["items"] == 1


def test_observed_counts_speakers_counts_distinct_speaker_id() -> None:
    """MIG-01f: speakers is now a real speaker_id-based count, not the old
    lect-based proxy — two items sharing a lect but with distinct
    speaker_ids count as 2 speakers (would have counted as 1 under the old
    proxy), and two items sharing a speaker_id count as 1."""
    items = [
        _item("a", speaker_id="spk-1", lect="Lafourche"),
        _item("b", speaker_id="spk-2", lect="Lafourche"),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["speakers"] == 2

    same_speaker_items = [
        _item("c", speaker_id="spk-3"),
        _item("d", speaker_id="spk-3"),
    ]
    same_speaker_scorecard = compute_coverage_scorecard(
        "frc", same_speaker_items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert same_speaker_scorecard.observed_counts["speakers"] == 1


def test_observed_counts_speakers_excludes_null_speaker_id() -> None:
    items = [_item("a", eligible=False)]  # eligible=False -> speaker_id=None
    scorecard = compute_coverage_scorecard(
        "lou", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["speakers"] == 0


# --- diff_catalog_coverage / unmet_cells --------------------------------------


def test_diff_catalog_coverage_reflects_each_cells_own_status() -> None:
    cells = [
        _cell("A-001", "high", "unmet"),
        _cell("B-001", "medium", "met"),
        _cell("C-001", "low", "partial"),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), cells,
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.diff_catalog_coverage == {"A-001": "unmet", "B-001": "met", "C-001": "partial"}


def test_unmet_cells_contains_only_unmet_status_cells() -> None:
    cells = [
        _cell("A-001", "high", "unmet"),
        _cell("B-001", "medium", "met"),
        _cell("C-001", "low", "partial"),
        _cell("D-001", "critical", "unmet"),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), cells,
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.unmet_cells == ["A-001", "D-001"]


def test_met_and_partial_cells_excluded_from_priorities() -> None:
    cells = [
        _cell("A-001", "critical", "met"),
        _cell("B-001", "critical", "partial"),
        _cell("C-001", "high", "unmet"),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), cells,
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.next_collection_priorities == ["C-001"]


# --- next_collection_priorities ordering --------------------------------------


def test_priorities_ordered_critical_high_medium_low() -> None:
    cells = [
        _cell("LOW-001", "low", "unmet"),
        _cell("MED-001", "medium", "unmet"),
        _cell("CRIT-001", "critical", "unmet"),
        _cell("HIGH-001", "high", "unmet"),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), cells,
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.next_collection_priorities == ["CRIT-001", "HIGH-001", "MED-001", "LOW-001"]


def test_priorities_break_ties_by_id() -> None:
    cells = [
        _cell("Z-001", "high", "unmet"),
        _cell("A-001", "high", "unmet"),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), cells,
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.next_collection_priorities == ["A-001", "Z-001"]


# --- pass-through fields -------------------------------------------------------


def test_annotation_hours_and_note_pass_through_unchanged() -> None:
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), [],
        annotation_hours_committed=12.5, annotation_hours_budget_note="set by PM",
    )
    assert scorecard.annotation_hours_committed == 12.5
    assert scorecard.annotation_hours_budget_note == "set by PM"


def test_targets_pass_through_unchanged() -> None:
    targets = _targets()
    scorecard = compute_coverage_scorecard(
        "frc", [], targets, [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.targets == targets


# --- floor_verdicts / aspirational_verdicts (issue #18) ------------------------
#
# _targets() is floor={"items": 5, "speakers": 3}, aspirational={"items": 20,
# "speakers": 5}; PARTIAL_THRESHOLD_FRACTION is 0.5.


def test_floor_verdict_met_when_observed_meets_target() -> None:
    items = [_item(f"item-{i}", speaker_id=f"spk-{i}") for i in range(5)]  # 5 distinct items, meets floor.items=5
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["items"] == 5
    assert scorecard.floor_verdicts["items"] == "met"


def test_floor_verdict_met_when_observed_exceeds_target() -> None:
    items = [_item(f"item-{i}", speaker_id=f"spk-{i}") for i in range(8)]  # exceeds floor.items=5
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.floor_verdicts["items"] == "met"


def test_floor_verdict_partial_when_observed_at_least_half_target() -> None:
    items = [_item(f"item-{i}", speaker_id=f"spk-{i}") for i in range(3)]  # 3/5 = 0.6 >= 0.5 threshold, < target
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.floor_verdicts["items"] == "partial"


def test_floor_verdict_unmet_when_observed_below_half_target() -> None:
    items = [_item("item-0")]  # 1/5 = 0.2 < 0.5 threshold
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.floor_verdicts["items"] == "unmet"


def test_floor_verdict_unmet_when_observed_zero() -> None:
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.floor_verdicts["items"] == "unmet"
    assert scorecard.floor_verdicts["speakers"] == "unmet"


def test_aspirational_verdict_computed_independently_of_floor() -> None:
    # 5 items meets floor.items=5 but is well below aspirational.items=20's
    # 0.5 threshold (10) -> unmet on the aspirational side.
    items = [_item(f"item-{i}", speaker_id=f"spk-{i}") for i in range(5)]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.floor_verdicts["items"] == "met"
    assert scorecard.aspirational_verdicts["items"] == "unmet"


def test_aspirational_verdict_partial() -> None:
    # 10/20 = 0.5, exactly at the partial threshold, below the target itself.
    items = [_item(f"item-{i}", speaker_id=f"spk-{i}") for i in range(10)]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.aspirational_verdicts["items"] == "partial"


def test_target_verdicts_cover_every_target_key() -> None:
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert set(scorecard.floor_verdicts) == {"items", "speakers"}
    assert set(scorecard.aspirational_verdicts) == {"items", "speakers"}


def test_target_verdict_only_counts_eligible_target_language_items() -> None:
    # Mirrors the observed_counts eligibility/language filtering tests above:
    # verdicts derive from the same filtered observed_counts, not raw items.
    items = [_item("a", eligible=True), _item("b", eligible=False), _item("c", language_tag="lou")]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["items"] == 1
    assert scorecard.floor_verdicts["items"] == "unmet"  # 1/5 < 0.5 threshold


# --- determinism ---------------------------------------------------------------


def test_compute_coverage_scorecard_is_deterministic() -> None:
    cells = [_cell("A-001", "high", "unmet")]
    items = [_item("a")]
    first = compute_coverage_scorecard(
        "frc", items, _targets(), cells,
        annotation_hours_committed=1.0, annotation_hours_budget_note="n/a",
    )
    second = compute_coverage_scorecard(
        "frc", items, _targets(), cells,
        annotation_hours_committed=1.0, annotation_hours_budget_note="n/a",
    )
    assert first == second


# --- schema_version (MIG-01f: default bumped to 2.0.0) -------------------------


def test_real_config_schema_version_is_2_0_0() -> None:
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.schema_version == "2.0.0"


# --- gate_class / base_failure_rate maps (MIG-01f) ------------------------------


def test_scorecard_gate_class_map_reflects_cell_values() -> None:
    cells = [
        _cell("VERB-001", "critical", "unmet", gate_class=1),
        _cell("PRO-002", "critical", "unmet", gate_class=1),
        _cell("DET-002", "high", "unmet", gate_class=2),
        _cell("NO-GATE-001", "low", "met", gate_class=None),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), cells,
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.gate_class == {"VERB-001": 1, "PRO-002": 1, "DET-002": 2}
    assert "NO-GATE-001" not in scorecard.gate_class


def test_scorecard_base_failure_rate_map_omits_none_valued_cells() -> None:
    cells = [
        _cell("VERB-001", "critical", "unmet", base_failure_rate=0.82),
        _cell("PRO-002", "critical", "unmet", base_failure_rate=0.61),
        _cell("NO-RATE-001", "low", "met", base_failure_rate=None),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", [], _targets(), cells,
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.base_failure_rate == {"VERB-001": 0.82, "PRO-002": 0.61}
    assert "NO-RATE-001" not in scorecard.base_failure_rate
    # Value types stay strictly float — no None/null placeholder entries.
    assert all(isinstance(v, float) for v in scorecard.base_failure_rate.values())


# --- stratified_counts (MIG-01f) ------------------------------------------------


def test_stratified_counts_by_speaker_generation() -> None:
    items = [
        _item("a", speaker_id="spk-1", speaker_generation="elder_fluent"),
        _item("b", speaker_id="spk-2", speaker_generation="elder_fluent"),
        _item("c", speaker_id="spk-3", speaker_generation="heritage"),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
        stratify_by=("speaker_generation",),
    )
    assert scorecard.stratified_counts["speaker_generation:elder_fluent"] == {"items": 2, "speakers": 2}
    assert scorecard.stratified_counts["speaker_generation:heritage"] == {"items": 1, "speakers": 1}
    # observed_counts' own top-level shape is unaffected by stratify_by.
    assert scorecard.observed_counts["items"] == 3


def test_stratified_counts_empty_when_stratify_by_not_given() -> None:
    items = [_item("a")]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.stratified_counts == {}


def test_stratified_counts_by_lect_and_genre_independently() -> None:
    items = [
        _item("a", lect="Lafourche", genre="conversation"),
        _item("b", lect="Terrebonne", genre="interview", speaker_id="spk-2"),
    ]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
        stratify_by=("lect", "genre"),
    )
    assert scorecard.stratified_counts["lect:Lafourche"] == {"items": 1, "speakers": 1}
    assert scorecard.stratified_counts["lect:Terrebonne"] == {"items": 1, "speakers": 1}
    assert scorecard.stratified_counts["genre:conversation"] == {"items": 1, "speakers": 1}
    assert scorecard.stratified_counts["genre:interview"] == {"items": 1, "speakers": 1}


# --- load_diff_catalog_cells: real committed files ------------------------------


def test_load_diff_catalog_cells_on_real_frc_file() -> None:
    cells = load_diff_catalog_cells(REAL_FRC_CATALOG)
    ids = {c.id for c in cells}
    assert ids == {
        "ASP-001", "PRO-002", "VERB-001", "PRO-003", "PRO-004", "DET-002",
        "AUX-001", "LEX-001", "LOAN-001", "CS-001",
    }


def test_load_diff_catalog_cells_reads_priority_and_status() -> None:
    cells = load_diff_catalog_cells(REAL_FRC_CATALOG)
    pro_002 = next(c for c in cells if c.id == "PRO-002")
    assert pro_002.priority == "critical"
    assert pro_002.coverage_status == "unmet"


def test_load_diff_catalog_cells_frc_vs_fra_keeps_surface_forms_and_failure_mode() -> None:
    """issue #18: the previous 5-field DiffCatalogCell dropped frc_form/
    fra_contrast/failure_mode/source that red-team (D3) and augment (D6)
    need — verify they now round-trip from the real file."""
    cells = load_diff_catalog_cells(REAL_FRC_CATALOG)
    pro_002 = next(c for c in cells if c.id == "PRO-002")
    assert pro_002.frc_form == "on has largely replaced nous, reinforced by an emphatic \"nous-autres\" form"
    assert pro_002.fra_contrast == "nous"
    assert pro_002.failure_mode == "systematic pronoun-frequency mismatch vs. Standard French training data"
    assert pro_002.source == "linguistics-deep-dive.md"
    # frc_vs_fra.yml cells never carry lou_form/note/anti-conflation fields.
    assert pro_002.lou_form is None
    assert pro_002.note is None
    assert pro_002.shared_surface_form is None
    assert pro_002.why_convergent_not_derived is None


def test_load_diff_catalog_cells_frc_vs_fra_verb_001_is_corrected() -> None:
    """MIG-01f: VERB-001 corrected to tech-spec v2 §4's present-tense-3pl
    -ont description, replacing the v1 draft's vaguer -aient contrast."""
    cells = load_diff_catalog_cells(REAL_FRC_CATALOG)
    verb_001 = next(c for c in cells if c.id == "VERB-001")
    assert verb_001.feature == "present-tense 3pl -ont"
    assert verb_001.frc_form == "ils mangeont"
    assert verb_001.fra_contrast == "ils mangent"
    assert verb_001.failure_mode == "regularized to Standard French -ent"
    assert verb_001.gate_class == 1
    assert verb_001.ortho_visible is True
    assert verb_001.modality == "text"
    assert verb_001.lect_scope == ("all",)
    assert verb_001.probe_task == ("normalize", "contrast", "judge")


def test_load_diff_catalog_cells_frc_vs_fra_has_no_fut_001() -> None:
    """MIG-01f: FUT-001 (frequency only) is dropped, not merely
    deprioritized, per tech-spec v2 §4."""
    cells = load_diff_catalog_cells(REAL_FRC_CATALOG)
    assert not any(c.id == "FUT-001" for c in cells)


def test_load_diff_catalog_cells_frc_vs_fra_has_new_v2_cells() -> None:
    cells = load_diff_catalog_cells(REAL_FRC_CATALOG)
    ids = {c.id for c in cells}
    assert {"PRO-004", "DET-002", "AUX-001", "LEX-001", "CS-001"} <= ids


def test_load_diff_catalog_cells_on_real_lou_vs_fra_file() -> None:
    cells = load_diff_catalog_cells(REAL_LOU_VS_FRA_CATALOG)
    assert len(cells) == 7
    ids = {c.id for c in cells}
    assert ids == {"TMA-001", "COP-001", "NEG-001", "DET-001", "PRO-001", "ADJ-001", "ORTH-001"}


def test_load_diff_catalog_cells_lou_vs_fra_keeps_lou_form_and_note() -> None:
    """lou_vs_fra.yml's extra `note` field (only ORTH-001 carries one) —
    previously unexercised by any loader test (issue #18)."""
    cells = load_diff_catalog_cells(REAL_LOU_VS_FRA_CATALOG)
    tma_001 = next(c for c in cells if c.id == "TMA-001")
    assert tma_001.lou_form is not None and "preverbal particles" in tma_001.lou_form
    assert tma_001.fra_contrast == "full inflectional conjugation"
    assert tma_001.note is None  # only ORTH-001 in this file carries a note

    orth_001 = next(c for c in cells if c.id == "ORTH-001")
    assert orth_001.note is not None
    assert "ORTH-001..010" in orth_001.note


def test_load_diff_catalog_cells_on_real_lou_vs_hat_file() -> None:
    """lou_vs_hat.yml's divergent shape (issue #18): no frc_form/lou_form/
    fra_contrast at all — instead shared_surface_form and
    why_convergent_not_derived, the anti-conflation-axis fields."""
    cells = load_diff_catalog_cells(REAL_LOU_VS_HAT_CATALOG)
    assert len(cells) == 1
    anti_hat_001 = cells[0]
    assert anti_hat_001.id == "ANTI-HAT-001"
    assert anti_hat_001.axis == "cross-linguistic-conflation"
    assert anti_hat_001.priority == "critical"
    assert anti_hat_001.coverage_status == "unmet"
    assert anti_hat_001.shared_surface_form is not None
    assert "TMA particles" in anti_hat_001.shared_surface_form
    assert anti_hat_001.note is not None
    # This file has no fra_contrast/frc_form/lou_form (hat is the contrast
    # language here, not fra) — confirms they're genuinely optional, not
    # silently required and coincidentally present in the other two files.
    assert anti_hat_001.fra_contrast is None
    assert anti_hat_001.frc_form is None
    assert anti_hat_001.lou_form is None


def test_load_diff_catalog_cells_lou_vs_hat_why_convergent_is_linguist_to_state() -> None:
    """MIG-01f: the v1 historical claim about Haitian-immigration timing is
    removed, replaced with the [LINGUIST TO STATE] placeholder, per
    tech-spec v2 §4."""
    cells = load_diff_catalog_cells(REAL_LOU_VS_HAT_CATALOG)
    anti_hat_001 = next(c for c in cells if c.id == "ANTI-HAT-001")
    assert anti_hat_001.why_convergent_not_derived == "[LINGUIST TO STATE]"
    assert anti_hat_001.gate_class == 1
    assert anti_hat_001.probe_task == ("judge", "translate_eng")


def test_load_diff_catalog_cells_on_real_frc_vs_fra_phon_file() -> None:
    """MIG-01f: new frc_vs_fra_phon.yml skeleton, P1-P13. P2 is the one cell
    tech-spec v2 §4 gives real worked-example content for."""
    cells = load_diff_catalog_cells(REAL_FRC_VS_FRA_PHON_CATALOG)
    ids = {c.id for c in cells}
    assert ids == {f"P{i}" for i in range(1, 14)}

    p2 = next(c for c in cells if c.id == "P2")
    assert p2.modality == "speech"
    assert p2.ortho_visible is False
    assert p2.environment == {"segment": "t d", "replacement": "ts dz", "following_context": "i y"}
    assert p2.respelling_attested == "true"
    assert p2.lect_scope == ("acadiana_west_prairie", "acadiana_central")
    assert p2.probe_task == ("phone_error", "g2p", "alignment")
    assert p2.gate_class is None
    assert p2.priority == "critical"

    # Every other cell (P1, P3-P13) is an engineering stub only.
    p1 = next(c for c in cells if c.id == "P1")
    assert p1.feature == "[LINGUIST TO STATE]"
    assert p1.modality == "speech"
    assert p1.ortho_visible is False
    assert p1.coverage_status == "unmet"


# --- lect_scope normalization (MIG-01f, user story 12) --------------------------


def test_lect_scope_normalizes_bare_string_and_list_to_tuple() -> None:
    bare_string_cell = DiffCatalogCell(
        id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
        lect_scope="all",  # type: ignore[arg-type]
    )
    assert bare_string_cell.lect_scope == ("all",)

    list_cell = DiffCatalogCell(
        id="X-002", axis="a", feature="f", priority="high", coverage_status="unmet",
        lect_scope=("acadiana_west_prairie", "acadiana_central"),
    )
    assert list_cell.lect_scope == ("acadiana_west_prairie", "acadiana_central")

    # Real-file round trip: VERB-001 spells lect_scope as a bare "all" string
    # in the YAML; PRO-002/P2 confirm a list also round-trips as a tuple.
    verb_001 = next(c for c in load_diff_catalog_cells(REAL_FRC_CATALOG) if c.id == "VERB-001")
    assert isinstance(verb_001.lect_scope, tuple)
    assert verb_001.lect_scope == ("all",)

    p2 = next(c for c in load_diff_catalog_cells(REAL_FRC_VS_FRA_PHON_CATALOG) if c.id == "P2")
    assert isinstance(p2.lect_scope, tuple)


# --- validation: v2 cell fields (gate_class, modality, probe_task, etc.) --------


def test_diff_catalog_cell_accepts_gate_class_0_through_3() -> None:
    for value in (0, 1, 2, 3):
        cell = DiffCatalogCell(
            id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
            gate_class=value,
        )
        assert cell.gate_class == value


def test_diff_catalog_cell_rejects_gate_class_outside_0_to_3() -> None:
    with pytest.raises(CoresetConfigError):
        DiffCatalogCell(
            id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
            gate_class=4,
        )
    with pytest.raises(CoresetConfigError):
        DiffCatalogCell(
            id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
            gate_class=-1,
        )


def test_diff_catalog_cell_gate_class_none_is_valid() -> None:
    cell = DiffCatalogCell(
        id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
        gate_class=None,
    )
    assert cell.gate_class is None


def test_diff_catalog_cell_rejects_invalid_modality() -> None:
    with pytest.raises(DataContractError):
        DiffCatalogCell(
            id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
            modality="written",  # type: ignore[arg-type]
        )


def test_diff_catalog_cell_rejects_invalid_probe_task() -> None:
    with pytest.raises(DataContractError):
        DiffCatalogCell(
            id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
            probe_task=("not_a_real_task",),  # type: ignore[arg-type]
        )


def test_diff_catalog_cell_rejects_invalid_respelling_attested() -> None:
    with pytest.raises(DataContractError):
        DiffCatalogCell(
            id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
            respelling_attested="maybe",  # type: ignore[arg-type]
        )


def test_diff_catalog_cell_accepts_respelling_attested_three_states() -> None:
    for value in ("true", "false", "unknown"):
        cell = DiffCatalogCell(
            id="X-001", axis="a", feature="f", priority="high", coverage_status="unmet",
            respelling_attested=value,
        )
        assert cell.respelling_attested == value


# --- validation: bad values and missing keys (issue #15) -----------------------


def test_diff_catalog_cell_rejects_invalid_priority() -> None:
    # Direct construction surfaces data_contract's own validate_literal error;
    # load_diff_catalog_cells() (tested below) wraps this into CoresetConfigError.
    with pytest.raises(DataContractError):
        DiffCatalogCell(id="X-001", axis="a", feature="f", priority="urgent", coverage_status="unmet")  # type: ignore[arg-type]


def test_diff_catalog_cell_rejects_invalid_coverage_status() -> None:
    with pytest.raises(DataContractError):
        DiffCatalogCell(id="X-001", axis="a", feature="f", priority="high", coverage_status="done")  # type: ignore[arg-type]


def test_load_diff_catalog_cells_rejects_bad_priority_value(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text(
        "cells:\n"
        "  - id: X-001\n"
        "    axis: morphosyntax\n"
        "    feature: test\n"
        "    priority: urgent\n"
        "    coverage_status: unmet\n",
        encoding="utf-8",
    )
    with pytest.raises(CoresetConfigError, match="priority"):
        load_diff_catalog_cells(path)


def test_load_diff_catalog_cells_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text(
        "cells:\n"
        "  - id: X-001\n"
        "    axis: morphosyntax\n"
        "    feature: test\n"
        "    priority: high\n",
        encoding="utf-8",
    )
    with pytest.raises(CoresetConfigError, match="coverage_status"):
        load_diff_catalog_cells(path)


# --- scorecard JSON round-trip (issue #18; tech-spec §8 "JSON, machine-readable") --


def test_scorecard_to_dict_is_json_serializable() -> None:
    cells = [_cell("A-001", "high", "unmet")]
    scorecard = compute_coverage_scorecard(
        "frc", [_item("a")], _targets(), cells,
        annotation_hours_committed=1.5, annotation_hours_budget_note="set by PM",
    )
    data = scorecard_to_dict(scorecard)
    # json.dumps must not raise: every field is already JSON-primitive.
    json.dumps(data)
    assert data["language"] == "frc"
    assert data["floor_verdicts"]["items"] == "unmet"


def test_scorecard_items_key_replaces_types_key() -> None:
    scorecard = compute_coverage_scorecard(
        "frc", [_item("a")], _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    data = scorecard_to_dict(scorecard)
    assert "items" in data["observed_counts"]
    assert "types" not in data["observed_counts"]
    assert "items" in data["floor_verdicts"]
    assert "types" not in data["floor_verdicts"]


def test_scorecard_to_json_and_from_json_round_trip() -> None:
    cells = [_cell("A-001", "high", "unmet"), _cell("B-001", "critical", "met")]
    original = compute_coverage_scorecard(
        "frc", [_item("a"), _item("b", speaker_id="spk-2")], _targets(), cells,
        annotation_hours_committed=2.0, annotation_hours_budget_note="set by PM",
    )
    text = scorecard_to_json(original)
    restored = scorecard_from_json(text)
    assert restored == original


def test_scorecard_to_dict_and_from_dict_round_trip() -> None:
    scorecard = compute_coverage_scorecard(
        "lou", [], _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    restored = scorecard_from_dict(scorecard_to_dict(scorecard))
    assert restored == scorecard


# --- read_manifest -> compute_coverage_scorecard (issue #18: real calling convention) --


def test_compute_coverage_scorecard_consumes_read_manifest_output(tmp_path: Path) -> None:
    """The exact broken chain R10 named: preprocess writes a manifest,
    DataItem(**row) used to raise TypeError on it, so compute_coverage_scorecard
    had never been fed real preprocess output. This feeds read_manifest's
    output straight into compute_coverage_scorecard, the real calling
    convention downstream tooling will use — now against the v2 36-field
    DataItem shape with speaker_id populated (MIG-01f)."""
    rows = []
    for i in range(5):
        rights = "cc_open"
        training_permission = "yes_general"
        consent = "informed_consent_training"
        cultural_sensitivity = "open"
        community_review_signed_off = False
        release_class = derive_release_class(
            ReleaseClassInputs(
                rights=rights,  # type: ignore[arg-type]
                training_permission=training_permission,  # type: ignore[arg-type]
                consent=consent,  # type: ignore[arg-type]
                cultural_sensitivity=cultural_sensitivity,  # type: ignore[arg-type]
                community_review_signed_off=community_review_signed_off,
            )
        )
        sensitivity_tier = "S0"
        pii_status = "none"
        cloud_ok = derive_cloud_ok(
            release_class=release_class,
            training_permission=training_permission,  # type: ignore[arg-type]
            sensitivity_tier=sensitivity_tier,  # type: ignore[arg-type]
            pii_status=pii_status,  # type: ignore[arg-type]
        )
        rows.append(
            {
                "item_id": f"item-{i}",
                "source": "test-collection",
                "record_type": "text",
                "language_tag": "frc",
                "eng_dialect": None,
                "lect": "Lafourche",
                "orthography_system": "ad_hoc",
                "genre": "conversation",
                "register": "casual",
                "rights": rights,
                "consent": consent,
                "training_permission": training_permission,
                "cultural_sensitivity": cultural_sensitivity,
                "community_review_signed_off": community_review_signed_off,
                "sensitivity_tier": sensitivity_tier,
                "access_tier": 1,
                "object_tier": "T0",
                "release_class": release_class,
                "speaker_id": f"spk-{i}",
                "speaker_generation": "elder_fluent",
                "speaker_role": "interviewee",
                "gender": "other_unknown",
                "attribution_mode": "anonymous",
                "pii_status": pii_status,
                "reading_type": None,
                "passage_id": None,
                "pair_id": None,
                "split": "silver_unreviewed",
                "data_class": "gold",
                "synthetic": False,
                "generator": None,
                "provenance": "original",
                "normalizer_status": "not_ready",
                "normalization_difficulty": "low",
                "diff_catalog_flags": [],
                "cloud_ok": cloud_ok,
                "schema_version": "2.0.0",
                # extra preprocess-only key read_manifest must drop:
                "code_switch_spans": [{"start": 0, "end": 4, "language_tag": "eng"}],
            }
        )
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    items = read_manifest(manifest_path)
    assert len(items) == 5

    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["items"] == 5
    assert scorecard.observed_counts["speakers"] == 5
    assert scorecard.floor_verdicts["items"] == "met"
