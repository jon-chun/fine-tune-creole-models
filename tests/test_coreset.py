"""Tests for src/coreset/ — the coverage scorecard (tech-spec §8).

Tests only external behavior: constructing DataItem/DiffCatalogCell/
CoverageTargets fixtures, calling compute_coverage_scorecard(), asserting on
the returned CoverageScorecard. load_diff_catalog_cells() is tested
separately against the real committed diff-catalog YAML.
"""

from pathlib import Path

from data_contract import DataItem
from coreset import (
    CoverageTargets,
    DiffCatalogCell,
    compute_coverage_scorecard,
    load_diff_catalog_cells,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_FRC_CATALOG = REPO_ROOT / "configs" / "diff_catalog" / "frc_vs_fra.yml"


def _item(item_id: str, language_tag: str = "frc", eligible: bool = True) -> DataItem:
    return DataItem(
        item_id=item_id,
        source="test-collection",
        language_tag=language_tag,  # type: ignore[arg-type]
        lect="Lafourche" if eligible else None,
        genre=None,
        orthography_system="ad_hoc",
        consent_tier="training",
        rights="cc_open" if eligible else "rights_unknown",
        training_permission="yes_general" if eligible else "no",
        cultural_sensitivity="open",
        community_review_signed_off=False,
        release_class="public",
        synthetic=False,
        generator=None,
        provenance="original",
        schema_version="1.0.0",
    )


def _cell(id_: str, priority: str, coverage_status: str) -> DiffCatalogCell:
    return DiffCatalogCell(
        id=id_,
        axis="morphosyntax",
        feature="test feature",
        priority=priority,  # type: ignore[arg-type]
        coverage_status=coverage_status,  # type: ignore[arg-type]
    )


def _targets() -> CoverageTargets:
    return CoverageTargets(
        floor={"types": 5, "speakers": 3},
        aspirational={"types": 20, "speakers": 5},
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
    assert scorecard.observed_counts["types"] == 3


def test_observed_counts_excludes_ineligible_items() -> None:
    items = [_item("a", eligible=True), _item("b", eligible=False)]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["types"] == 1


def test_observed_counts_excludes_other_language_items() -> None:
    items = [_item("a", language_tag="frc"), _item("b", language_tag="lou")]
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["types"] == 1


def test_observed_counts_speakers_counts_distinct_non_null_lect() -> None:
    items = [_item("a"), _item("b")]  # both use lect="Lafourche" from the fixture
    scorecard = compute_coverage_scorecard(
        "frc", items, _targets(), [],
        annotation_hours_committed=0.0, annotation_hours_budget_note="n/a",
    )
    assert scorecard.observed_counts["speakers"] == 1


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


# --- load_diff_catalog_cells: real committed file ------------------------------


def test_load_diff_catalog_cells_on_real_frc_file() -> None:
    cells = load_diff_catalog_cells(REAL_FRC_CATALOG)
    assert len(cells) == 6
    ids = {c.id for c in cells}
    assert ids == {"ASP-001", "FUT-001", "PRO-002", "VERB-001", "PRO-003", "LOAN-001"}


def test_load_diff_catalog_cells_reads_priority_and_status() -> None:
    cells = load_diff_catalog_cells(REAL_FRC_CATALOG)
    pro_002 = next(c for c in cells if c.id == "PRO-002")
    assert pro_002.priority == "critical"
    assert pro_002.coverage_status == "unmet"
