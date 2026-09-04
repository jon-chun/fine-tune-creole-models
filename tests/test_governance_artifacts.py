"""Tests for src/governance/artifacts.py — the datasheet, model-card, and
language-readiness-evidence generators (tech-spec v2 §7; backlog 0016).

Tests only external behavior: constructing DataItem/ConsentLedgerEntry/
RunMetadata/CandidateResult/BakeoffRunResult/CoverageScorecard fixtures
directly (never via run_bakeoff/compute_coverage_scorecard — those are
sibling modules' own orchestration, already covered by their own test
files), calling build_datasheet/build_model_card/build_readiness_evidence/
write_artifacts, and asserting on the returned dataclasses, to_dict()
output, raised errors, or on-disk file shape.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bakeoff import (
    BakeoffRunResult,
    CandidateResult,
    GateClass,
    RedTeamCellResult,
    RedTeamVerdict,
    ScoreResult,
    SeedAggregatedScore,
    derive_model_release_class,
)
from data_contract import (
    DataItem,
    ReleaseClassInputs,
    derive_cloud_ok,
    derive_release_class,
)
from eval import AcceptanceReport, Layer
from coreset import CoverageScorecard, CoverageTargets, TargetVerdict
from governance import ConsentLedgerEntry, GovernanceError, append_entry
from tracking import RunMetadata, to_hashable_config

from governance.artifacts import (
    HAT_UNTESTABLE_STATEMENT,
    READINESS_COVERAGE_MET_FRACTION,
    READINESS_COVERAGE_PARTIAL_FRACTION,
    READINESS_RELEASE_CLASSES_FOR_GO,
    READINESS_RELEASE_CLASSES_FOR_PARTIAL,
    LicenseLineage,
    LouMtArmResult,
    LouMtBakeoffResult,
    build_datasheet,
    build_model_card,
    build_readiness_evidence,
    lineage_clear,
    write_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_REPORT_EXAMPLE_PATH = REPO_ROOT / "tests" / "fixtures" / "governance" / "eval_report_example.json"

_T0 = datetime(2026, 1, 1, 12, 0, 0)


# --- Fixtures ----------------------------------------------------------------


def _item(
    item_id: str,
    *,
    speaker_id: str | None = "spk-1",
    consent: str = "informed_consent_training",
    training_permission: str = "yes_general",
    data_class: str = "gold",
    provenance: str = "original",
    rights: str = "cc_open",
) -> DataItem:
    """A real v2 DataItem fixture, built the same way tests/test_coreset.py's
    own `_item` helper does: rights/training_permission/consent chosen so
    the derived release_class/cloud_ok fields never disagree with a
    hand-set value (DataItem.__post_init__ would raise otherwise)."""
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
        language_tag="frc",
        eng_dialect=None,
        lect="Lafourche",
        orthography_system="ad_hoc",
        genre="conversation",
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
        speaker_id=speaker_id,
        speaker_generation="elder_fluent",
        speaker_role="interviewee",
        gender="other_unknown",
        attribution_mode="anonymous",
        pii_status=pii_status,  # type: ignore[arg-type]
        reading_type=None,
        passage_id=None,
        pair_id=None,
        split="silver_unreviewed",
        data_class=data_class,  # type: ignore[arg-type]
        synthetic=False,
        generator=None,
        provenance=provenance,
        normalizer_status="not_ready",
        normalization_difficulty="low",
        diff_catalog_flags=[],
        cloud_ok=cloud_ok,
        schema_version="2.0.0",
    )


def _run_metadata(**overrides: object) -> RunMetadata:
    defaults: dict[str, object] = dict(
        run_id="frc-lucie7b-s1",
        stage="eval",
        language="frc",
        config=to_hashable_config({"candidate_id": "lucie-7b"}),
        git_commit_sha="abc1234",
        started_at=_T0,
        completed_at=_T0 + timedelta(hours=1),
        artifact_refs=("adapters/lucie-7b/seed-1/",),
        seed=1,
        split_id="split-2026-01-01",
        lock_hash="deadbeef",
        tree_dirty=False,
        gpu_hours=1.2,
        usd=4.05,
        instance="H100-80GB",
        manifest_sha256="manifest-hash-abc",
    )
    defaults.update(overrides)
    return RunMetadata(**defaults)  # type: ignore[arg-type]


def _cell(
    cell_id: str = "VERB-001",
    *,
    class_assigned: GateClass = 3,
    base_rate: float = 0.8,
    tuned_rate: float = 0.1,
) -> RedTeamCellResult:
    return RedTeamCellResult(
        cell_id=cell_id,
        base_rate=base_rate,
        tuned_rate=tuned_rate,
        wilson_95=(0.05, 0.15),
        mcnemar_p=0.001,
        gate_class=class_assigned,
        class_assigned=class_assigned,
    )


def _verdict(class_assigned: GateClass = 3) -> RedTeamVerdict:
    return RedTeamVerdict(
        probe_set_version="v2026-09-03", cells={"VERB-001": _cell(class_assigned=class_assigned)}
    )


def _candidate_result(
    candidate_id: str,
    *,
    class_assigned: GateClass = 3,
    license_lineage_clear: bool = True,
    score_mean: float = 0.9,
) -> CandidateResult:
    verdict = _verdict(class_assigned=class_assigned)
    release_class = derive_model_release_class(verdict, license_lineage_clear=license_lineage_clear)
    return CandidateResult(
        candidate_id=candidate_id,
        untuned_base_score=ScoreResult("gold_accuracy", 0.4, higher_is_better=True),
        score=SeedAggregatedScore(
            metric_name="gold_accuracy",
            mean=score_mean,
            spread=0.02,
            higher_is_better=True,
            per_seed=(ScoreResult("gold_accuracy", score_mean, higher_is_better=True),),
        ),
        red_team_verdict=verdict,
        disqualified=verdict.disqualified,
        release_class=release_class,
        raw_metrics={"perplexity": 12.3},
    )


def _bakeoff_result(
    language: str = "frc",
    *,
    winner_id: str | None = "lucie-7b",
    results: tuple[CandidateResult, ...] | None = None,
) -> BakeoffRunResult:
    if results is None:
        results = (_candidate_result("lucie-7b"), _candidate_result("mistral-7b-v0.3", score_mean=0.5))
    return BakeoffRunResult(language=language, results=results, winner_candidate_id=winner_id)  # type: ignore[arg-type]


def _acceptance_report(
    *,
    forgetting_axis_flagged: bool = False,
    forgetting_axis_delta: float | None = -0.7,
) -> AcceptanceReport:
    return AcceptanceReport(
        language="frc",
        metrics={"chrf": 55.0},
        item_count=40,
        release_class="research_only",
        layer="B",
        forgetting_axis_flagged=forgetting_axis_flagged,
        forgetting_axis_delta=forgetting_axis_delta,
    )


def _eval_report_example() -> dict[str, object]:
    """The utils-spec benchmark v2 §7 example, loaded from the pinned
    fixture file (verbatim copy of the JSON in
    dev/utils-spec_fine-tune-cajun-benchmark_v2_fable51max_20260902.md
    §7), with `run_id`/`manifest_sha256` overridden to match this test
    module's own `_run_metadata()` fixture so the run_id/manifest_sha256
    join checks pass by default; individual tests override further as
    needed."""
    report = json.loads(EVAL_REPORT_EXAMPLE_PATH.read_text(encoding="utf-8"))
    report["run_id"] = "frc-lucie7b-s1"
    report["manifest_sha256"] = "manifest-hash-abc"
    return report  # type: ignore[no-any-return]


def _license_lineage(*, clear: bool = True) -> LicenseLineage:
    if clear:
        return LicenseLineage(
            base_license="apache-2.0", adapter_license="apache-2.0", data_licenses=("cc-by-4.0",)
        )
    return LicenseLineage(
        base_license="apache-2.0", adapter_license="cc-by-nc-sa-4.0", data_licenses=("cc-by-4.0",)
    )


_RELEASE_LICENSES = ("apache-2.0", "mit", "cc-by-4.0")


def _targets() -> CoverageTargets:
    return CoverageTargets(floor={"items": 10, "speakers": 2}, aspirational={"items": 20, "speakers": 5})


def _scorecard(
    *,
    language: str = "frc",
    floor_verdicts: dict[str, TargetVerdict] | None = None,
    unmet_cells: list[str] | None = None,
) -> CoverageScorecard:
    return CoverageScorecard(
        language=language,  # type: ignore[arg-type]
        schema_version="2.0.0",
        targets=_targets(),
        observed_counts={"items": 10, "speakers": 2},
        floor_verdicts=floor_verdicts if floor_verdicts is not None else {"items": "met", "speakers": "met"},
        aspirational_verdicts={"items": "partial", "speakers": "partial"},
        diff_catalog_coverage={"VERB-001": "unmet"},
        gate_class={"VERB-001": 1},
        base_failure_rate={"VERB-001": 0.82},
        unmet_cells=unmet_cells if unmet_cells is not None else ["VERB-001"],
        next_collection_priorities=["VERB-001"],
        annotation_hours_committed=0.0,
        annotation_hours_budget_note="test",
        stratified_counts={},
    )


def _lou_mt_result(*, winner_arm_id: str | None = "kreyol-mt") -> LouMtBakeoffResult:
    return LouMtBakeoffResult(
        arms=(
            LouMtArmResult(arm_id="kreyol-mt", chrf=45.0, bleu=20.0, is_no_creole_control=False),
            LouMtArmResult(arm_id="french-native-adapter", chrf=42.0, bleu=18.0, is_no_creole_control=False),
            LouMtArmResult(arm_id="mbart-50-no-creole", chrf=30.0, bleu=10.0, is_no_creole_control=True),
        ),
        winner_arm_id=winner_arm_id,
        tie_break_applied=False,
    )


# --- Datasheet -----------------------------------------------------------------


def test_datasheet_lists_consent_and_training_permission_per_item() -> None:
    items = [
        _item("item-1", consent="informed_consent_training", training_permission="yes_general"),
        _item("item-2", consent="legacy_no_consent", training_permission="uncertain"),
    ]
    ledger: list[ConsentLedgerEntry] = []
    datasheet = build_datasheet(
        items,
        ledger=ledger,
        run_metadata=_run_metadata(),
        dataset_id="frc-corpus-v1",
        collection_method="field recordings",
        annotator_info="two native-speaker linguists",
        known_limitations=("small n",),
    )
    rows_by_id = {row["item_id"]: row for row in datasheet.items}
    assert rows_by_id["item-1"]["consent"] == "informed_consent_training"
    assert rows_by_id["item-1"]["training_permission"] == "yes_general"
    assert rows_by_id["item-2"]["consent"] == "legacy_no_consent"
    assert rows_by_id["item-2"]["training_permission"] == "uncertain"
    assert rows_by_id["item-1"]["rights"] == "cc_open"
    assert rows_by_id["item-1"]["data_class"] == "gold"
    assert rows_by_id["item-1"]["provenance"] == "original"
    assert datasheet.item_count == 2
    assert datasheet.data_class_counts["gold"] == 2


def test_datasheet_prefers_ledger_current_consent_over_item_field() -> None:
    """A ledger entry withdrawing consent after ingestion must win over the
    DataItem's own (stale) consent field — the ledger is the "what applies
    right now" source (governance.current_consent's own contract)."""
    items = [_item("item-1", consent="informed_consent_training", training_permission="yes_general")]
    ledger = append_entry(
        [],
        ConsentLedgerEntry(
            item_id="item-1",
            speaker_id="spk-1",
            consent="consent_withdrawn",
            training_permission="no",
            granted_at=_T0 + timedelta(days=1),
            source_note="withdrawal",
        ),
    )
    datasheet = build_datasheet(
        items,
        ledger=ledger,
        run_metadata=_run_metadata(),
        dataset_id="frc-corpus-v1",
        collection_method="field recordings",
        annotator_info="two native-speaker linguists",
        known_limitations=(),
    )
    row = datasheet.items[0]
    assert row["consent"] == "consent_withdrawn"
    assert row["training_permission"] == "no"


def test_datasheet_to_dict_is_json_serializable_with_iso_datetimes() -> None:
    items = [_item("item-1")]
    datasheet = build_datasheet(
        items,
        ledger=[],
        run_metadata=_run_metadata(),
        dataset_id="frc-corpus-v1",
        collection_method="field recordings",
        annotator_info="two native-speaker linguists",
        known_limitations=("small n",),
    )
    payload = datasheet.to_dict()
    text = json.dumps(payload)  # raises TypeError if anything is non-JSON-safe
    reloaded = json.loads(text)
    assert reloaded["generated_at"] == datasheet.generated_at.isoformat()
    assert isinstance(reloaded["generated_at"], str)
    assert reloaded["dataset_id"] == "frc-corpus-v1"
    assert reloaded["known_limitations"] == ["small n"]


# --- Model card ------------------------------------------------------------


def test_model_card_release_class_is_derived_never_hand_set() -> None:
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b", class_assigned=3),))
    card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={"rank": 16},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        run_metadata=_run_metadata(),
        lineage=_license_lineage(clear=True),
        release_licenses=_RELEASE_LICENSES,
        intended_uses=("assistive translation review",),
        prohibited_uses=("unsupervised chatbot",),
        limitations=("small gold set",),
    )
    # Class 3 (minor) + clear lineage -> release_ready, per
    # derive_model_release_class's own rule (worst_class >= 2 and license
    # lineage clear); never hand-set on ModelCard itself.
    assert card.release_class == "release_ready"


def test_model_card_nc_lineage_blocks_release_ready() -> None:
    bakeoff_result = _bakeoff_result(
        results=(_candidate_result("lucie-7b", class_assigned=3, license_lineage_clear=False),)
    )
    card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={"rank": 16},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        run_metadata=_run_metadata(),
        lineage=_license_lineage(clear=False),  # NC adapter license
        release_licenses=_RELEASE_LICENSES,
        intended_uses=(),
        prohibited_uses=(),
        limitations=(),
    )
    assert card.release_class == "research_only"
    assert card.lineage_is_clear is False


def test_model_card_release_ready_with_unclear_lineage_raises() -> None:
    """derive_model_release_class can never itself produce release_ready
    with an unclear lineage, so this exercises build_model_card's
    defense-in-depth guard directly via a CandidateResult whose own
    (pre-computed) release_class disagrees with what a fresh derivation
    against an unclear lineage would produce — a directly inconsistent
    caller input, which must raise rather than silently trust the stale
    CandidateResult.release_class."""
    inconsistent_result = CandidateResult(
        candidate_id="lucie-7b",
        untuned_base_score=ScoreResult("gold_accuracy", 0.4, higher_is_better=True),
        score=SeedAggregatedScore(
            metric_name="gold_accuracy",
            mean=0.9,
            spread=0.0,
            higher_is_better=True,
            per_seed=(ScoreResult("gold_accuracy", 0.9, higher_is_better=True),),
        ),
        red_team_verdict=_verdict(class_assigned=3),
        disqualified=False,
        release_class="release_ready",  # stale/wrong: lineage below is unclear
        raw_metrics={},
    )
    bakeoff_result = _bakeoff_result(results=(inconsistent_result,))
    with pytest.raises(GovernanceError):
        build_model_card(
            bakeoff_result=bakeoff_result,
            candidate_id="lucie-7b",
            hyperparameters={},
            acceptance_reports={"B": _acceptance_report()},
            eval_report=_eval_report_example(),
            run_metadata=_run_metadata(),
            lineage=_license_lineage(clear=False),
            release_licenses=_RELEASE_LICENSES,
            intended_uses=(),
            prohibited_uses=(),
            limitations=(),
        )


def test_model_card_carries_bakeoff_table_for_all_candidates() -> None:
    bakeoff_result = _bakeoff_result(
        results=(
            _candidate_result("lucie-7b", class_assigned=3, score_mean=0.9),
            _candidate_result("mistral-7b-v0.3", class_assigned=1, score_mean=0.5),
        )
    )
    card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        run_metadata=_run_metadata(),
        lineage=_license_lineage(clear=True),
        release_licenses=_RELEASE_LICENSES,
        intended_uses=(),
        prohibited_uses=(),
        limitations=(),
    )
    candidate_ids = {row["candidate_id"] for row in card.bakeoff_table}
    assert candidate_ids == {"lucie-7b", "mistral-7b-v0.3"}
    disqualified_row = next(row for row in card.bakeoff_table if row["candidate_id"] == "mistral-7b-v0.3")
    assert disqualified_row["disqualified"] is True
    markdown = card.render_markdown()
    assert "lucie-7b" in markdown and "mistral-7b-v0.3" in markdown


def test_model_card_carries_base_vs_tuned_per_cell_table_from_eval_report() -> None:
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        run_metadata=_run_metadata(),
        lineage=_license_lineage(clear=True),
        release_licenses=_RELEASE_LICENSES,
        intended_uses=(),
        prohibited_uses=(),
        limitations=(),
    )
    assert len(card.per_cell_table) == 1
    row = card.per_cell_table[0]
    assert row["cell_id"] == "VERB-001"
    assert row["base_rate"] == 0.82
    assert row["tuned_rate"] == 0.35
    assert row["wilson_95"] == (0.21, 0.52)
    assert row["gate_class"] == 1
    assert row["class_assigned"] == 2
    assert card.forgetting_report["fra_matched_f1_delta"] == -0.7
    assert card.forgetting_axis_delta == -0.7


def test_model_card_reports_gpu_hours_usd_instance() -> None:
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        run_metadata=_run_metadata(gpu_hours=1.2, usd=4.05, instance="H100-80GB"),
        lineage=_license_lineage(clear=True),
        release_licenses=_RELEASE_LICENSES,
        intended_uses=(),
        prohibited_uses=(),
        limitations=(),
    )
    assert card.gpu_hours == 1.2
    assert card.usd == 4.05
    assert card.instance == "H100-80GB"
    markdown = card.render_markdown()
    assert "1.2" in markdown and "4.05" in markdown and "H100-80GB" in markdown


def test_model_card_falls_back_to_eval_report_cost_when_run_metadata_missing() -> None:
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    run_metadata = _run_metadata(gpu_hours=None, usd=None, instance=None)
    card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        run_metadata=run_metadata,
        lineage=_license_lineage(clear=True),
        release_licenses=_RELEASE_LICENSES,
        intended_uses=(),
        prohibited_uses=(),
        limitations=(),
    )
    assert card.gpu_hours == 1.2
    assert card.usd == 4.05
    assert card.instance == "H100-80GB"


def test_eval_report_nulls_render_as_not_available() -> None:
    report = _eval_report_example()
    cells = report["cells"]
    assert isinstance(cells, list)
    first_cell = cells[0]
    assert isinstance(first_cell, dict)
    first_cell["base_rate"] = None
    first_cell["tuned_rate"] = None
    report["cost"] = {"gpu_hours": None, "usd": None, "instance": None}
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=report,
        run_metadata=_run_metadata(gpu_hours=None, usd=None, instance=None),
        lineage=_license_lineage(clear=True),
        release_licenses=_RELEASE_LICENSES,
        intended_uses=(),
        prohibited_uses=(),
        limitations=(),
    )
    assert card.gpu_hours is None
    assert card.usd is None
    markdown = card.render_markdown()
    assert "not available" in markdown
    # JSON output keeps real None, never invents or stringifies it early.
    payload = card.to_dict()
    assert payload["gpu_hours"] is None
    json.dumps(payload)


def test_run_id_mismatch_raises() -> None:
    report = _eval_report_example()
    report["run_id"] = "some-other-run-id"
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    with pytest.raises(GovernanceError, match="run_id"):
        build_model_card(
            bakeoff_result=bakeoff_result,
            candidate_id="lucie-7b",
            hyperparameters={},
            acceptance_reports={"B": _acceptance_report()},
            eval_report=report,
            run_metadata=_run_metadata(),
            lineage=_license_lineage(clear=True),
            release_licenses=_RELEASE_LICENSES,
            intended_uses=(),
            prohibited_uses=(),
            limitations=(),
        )


def test_manifest_sha256_mismatch_raises() -> None:
    report = _eval_report_example()
    report["manifest_sha256"] = "a-different-hash"
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    with pytest.raises(GovernanceError, match="manifest_sha256"):
        build_model_card(
            bakeoff_result=bakeoff_result,
            candidate_id="lucie-7b",
            hyperparameters={},
            acceptance_reports={"B": _acceptance_report()},
            eval_report=report,
            run_metadata=_run_metadata(manifest_sha256="manifest-hash-abc"),
            lineage=_license_lineage(clear=True),
            release_licenses=_RELEASE_LICENSES,
            intended_uses=(),
            prohibited_uses=(),
            limitations=(),
        )


def test_manifest_sha256_none_on_either_side_is_tolerated_not_a_mismatch() -> None:
    report = _eval_report_example()
    report["manifest_sha256"] = None
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=report,
        run_metadata=_run_metadata(manifest_sha256="manifest-hash-abc"),
        lineage=_license_lineage(clear=True),
        release_licenses=_RELEASE_LICENSES,
        intended_uses=(),
        prohibited_uses=(),
        limitations=(),
    )
    assert card.manifest_sha256 == "manifest-hash-abc"


def test_model_card_unknown_candidate_id_raises() -> None:
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    with pytest.raises(GovernanceError, match="not-a-real-candidate"):
        build_model_card(
            bakeoff_result=bakeoff_result,
            candidate_id="not-a-real-candidate",
            hyperparameters={},
            acceptance_reports={"B": _acceptance_report()},
            eval_report=_eval_report_example(),
            run_metadata=_run_metadata(),
            lineage=_license_lineage(clear=True),
            release_licenses=_RELEASE_LICENSES,
            intended_uses=(),
            prohibited_uses=(),
            limitations=(),
        )


# --- lineage_clear -----------------------------------------------------------


def test_lineage_clear_true_when_every_component_in_allowlist() -> None:
    assert lineage_clear(_license_lineage(clear=True), release_licenses=_RELEASE_LICENSES) is True


def test_lineage_clear_false_when_any_component_nc() -> None:
    nc_lineage = LicenseLineage(
        base_license="apache-2.0", adapter_license="cc-by-nc-sa-4.0", data_licenses=()
    )
    assert lineage_clear(nc_lineage, release_licenses=_RELEASE_LICENSES) is False


def test_lineage_clear_is_case_and_whitespace_insensitive() -> None:
    lineage = LicenseLineage(base_license=" Apache-2.0 ", adapter_license="APACHE-2.0", data_licenses=())
    assert lineage_clear(lineage, release_licenses=_RELEASE_LICENSES) is True


# --- Readiness evidence ------------------------------------------------------


def test_readiness_lou_always_states_hat_untestable() -> None:
    bakeoff_result = _bakeoff_result(language="lou", results=(_candidate_result("french-native"),))
    evidence = build_readiness_evidence(
        language="lou",
        scorecard=_scorecard(language="lou"),
        bakeoff_result=bakeoff_result,
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        lou_mt_result=_lou_mt_result(),
    )
    assert evidence.hat_untestable_statement == HAT_UNTESTABLE_STATEMENT
    assert "hat" in evidence.hat_untestable_statement
    assert HAT_UNTESTABLE_STATEMENT in evidence.render_markdown()


def test_readiness_frc_has_no_hat_untestable_statement() -> None:
    bakeoff_result = _bakeoff_result(language="frc")
    evidence = build_readiness_evidence(
        language="frc",
        scorecard=_scorecard(language="frc"),
        bakeoff_result=bakeoff_result,
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        lou_mt_result=None,
    )
    assert evidence.hat_untestable_statement is None


def test_readiness_lou_requires_mt_result() -> None:
    bakeoff_result = _bakeoff_result(language="lou", results=(_candidate_result("french-native"),))
    with pytest.raises(GovernanceError, match="lou_mt_result"):
        build_readiness_evidence(
            language="lou",
            scorecard=_scorecard(language="lou"),
            bakeoff_result=bakeoff_result,
            acceptance_reports={"B": _acceptance_report()},
            eval_report=_eval_report_example(),
            lou_mt_result=None,
        )


def test_readiness_reports_lou_mt_bakeoff_result() -> None:
    bakeoff_result = _bakeoff_result(language="lou", results=(_candidate_result("french-native"),))
    mt_result = _lou_mt_result(winner_arm_id="kreyol-mt")
    evidence = build_readiness_evidence(
        language="lou",
        scorecard=_scorecard(language="lou"),
        bakeoff_result=bakeoff_result,
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        lou_mt_result=mt_result,
    )
    assert evidence.lou_mt_result is mt_result
    assert evidence.to_dict()["lou_mt_result"]["winner_arm_id"] == "kreyol-mt"
    markdown = evidence.render_markdown()
    assert "kreyol-mt" in markdown


def test_readiness_recommendation_rule_pinned() -> None:
    """Pins the documented go/partial/no_go rule against a fixed scorecard
    + winner release_class combination, so a future edit to the rule's
    thresholds or logic must consciously update this test rather than
    silently drift."""
    all_met_scorecard = _scorecard(floor_verdicts={"items": "met", "speakers": "met"})
    bakeoff_result = _bakeoff_result(
        results=(_candidate_result("lucie-7b", class_assigned=3),), winner_id="lucie-7b"
    )
    evidence = build_readiness_evidence(
        language="frc",
        scorecard=all_met_scorecard,
        bakeoff_result=bakeoff_result,
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        lou_mt_result=None,
    )
    assert evidence.winner_release_class == "release_ready"
    assert evidence.recommendation == "go"

    partial_scorecard = _scorecard(floor_verdicts={"items": "met", "speakers": "unmet"})
    research_only_result = _candidate_result("lucie-7b", class_assigned=1)  # Class 1 -> research_only
    bakeoff_result_partial = _bakeoff_result(
        results=(research_only_result,), winner_id="lucie-7b"
    )
    evidence_partial = build_readiness_evidence(
        language="frc",
        scorecard=partial_scorecard,
        bakeoff_result=bakeoff_result_partial,
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        lou_mt_result=None,
    )
    assert evidence_partial.winner_release_class == "research_only"
    assert evidence_partial.recommendation == "partial"

    no_go_scorecard = _scorecard(floor_verdicts={"items": "unmet", "speakers": "unmet"})
    evidence_no_go = build_readiness_evidence(
        language="frc",
        scorecard=no_go_scorecard,
        bakeoff_result=bakeoff_result_partial,
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        lou_mt_result=None,
    )
    assert evidence_no_go.recommendation == "no_go"

    # The rule's own documented thresholds are the module constants this
    # test exercises against — asserting on them directly pins the
    # Open-Q placeholder values themselves, not just the resulting labels.
    assert READINESS_COVERAGE_MET_FRACTION == 1.0
    assert READINESS_COVERAGE_PARTIAL_FRACTION == 0.5
    assert READINESS_RELEASE_CLASSES_FOR_GO == frozenset({"release_ready"})
    assert READINESS_RELEASE_CLASSES_FOR_PARTIAL == frozenset({"release_ready", "research_only"})


def test_readiness_no_go_is_reported_not_narrowed() -> None:
    """A no_go recommendation must still be a fully-populated
    ReadinessEvidence (coverage summary, unmet cells, winner fields all
    present) — never a narrowed-scope stub (tech-spec v2 §7, PRD §3)."""
    no_go_scorecard = _scorecard(floor_verdicts={"items": "unmet", "speakers": "unmet"})
    disqualified_result = _candidate_result("lucie-7b", class_assigned=0)
    bakeoff_result = BakeoffRunResult(
        language="frc", results=(disqualified_result,), winner_candidate_id=None
    )
    evidence = build_readiness_evidence(
        language="frc",
        scorecard=no_go_scorecard,
        bakeoff_result=bakeoff_result,
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        lou_mt_result=None,
    )
    assert evidence.recommendation == "no_go"
    assert evidence.winner_candidate_id is None
    assert evidence.winner_release_class is None
    assert evidence.unmet_cells == ("VERB-001",)
    assert evidence.coverage_summary["observed_items"] == 10
    markdown = evidence.render_markdown()
    assert "no_go" in markdown
    payload = evidence.to_dict()
    json.dumps(payload)


def test_readiness_language_mismatch_raises() -> None:
    bakeoff_result = _bakeoff_result(language="frc")
    with pytest.raises(GovernanceError, match="language"):
        build_readiness_evidence(
            language="lou",
            scorecard=_scorecard(language="lou"),
            bakeoff_result=bakeoff_result,  # language=frc, mismatched
            acceptance_reports={"B": _acceptance_report()},
            eval_report=_eval_report_example(),
            lou_mt_result=_lou_mt_result(),
        )


# --- write_artifacts ---------------------------------------------------------


def test_write_artifacts_filenames_match_tech_spec(tmp_path: Path) -> None:
    items = [_item("item-1")]
    datasheet = build_datasheet(
        items,
        ledger=[],
        run_metadata=_run_metadata(),
        dataset_id="frc-corpus-v1",
        collection_method="field recordings",
        annotator_info="linguist",
        known_limitations=(),
    )
    bakeoff_result = _bakeoff_result(results=(_candidate_result("lucie-7b"),))
    model_card = build_model_card(
        bakeoff_result=bakeoff_result,
        candidate_id="lucie-7b",
        hyperparameters={},
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        run_metadata=_run_metadata(),
        lineage=_license_lineage(clear=True),
        release_licenses=_RELEASE_LICENSES,
        intended_uses=(),
        prohibited_uses=(),
        limitations=(),
    )
    readiness = build_readiness_evidence(
        language="frc",
        scorecard=_scorecard(),
        bakeoff_result=bakeoff_result,
        acceptance_reports={"B": _acceptance_report()},
        eval_report=_eval_report_example(),
        lou_mt_result=None,
    )

    paths = write_artifacts(tmp_path, datasheet=datasheet, model_card=model_card, readiness=readiness)

    expected = {
        "datasheet_md": tmp_path / "datasheet_frc-corpus-v1.md",
        "datasheet_json": tmp_path / "datasheet_frc-corpus-v1.json",
        "model_card_md": tmp_path / "model_card_lucie-7b.md",
        "model_card_json": tmp_path / "model_card_lucie-7b.json",
        "readiness_evidence_md": tmp_path / "readiness_evidence_frc.md",
        "readiness_evidence_json": tmp_path / "readiness_evidence_frc.json",
    }
    assert paths == expected
    for path in expected.values():
        assert path.exists()

    # JSON files round-trip as real JSON.
    json.loads(expected["datasheet_json"].read_text(encoding="utf-8"))
    json.loads(expected["model_card_json"].read_text(encoding="utf-8"))
    json.loads(expected["readiness_evidence_json"].read_text(encoding="utf-8"))

    # Markdown files are non-empty and carry the artifact's own header.
    assert "Datasheet" in expected["datasheet_md"].read_text(encoding="utf-8")
    assert "Model card" in expected["model_card_md"].read_text(encoding="utf-8")
    assert "readiness" in expected["readiness_evidence_md"].read_text(encoding="utf-8").lower()


def test_write_artifacts_only_writes_supplied_artifacts(tmp_path: Path) -> None:
    datasheet = build_datasheet(
        [_item("item-1")],
        ledger=[],
        run_metadata=_run_metadata(),
        dataset_id="frc-corpus-v1",
        collection_method="field recordings",
        annotator_info="linguist",
        known_limitations=(),
    )
    paths = write_artifacts(tmp_path, datasheet=datasheet)
    assert set(paths) == {"datasheet_md", "datasheet_json"}
