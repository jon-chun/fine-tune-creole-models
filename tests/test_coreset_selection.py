"""Tests for src/coreset/selection.py — the coreset selection algorithm and
per-run manifest (tech-spec v2 §8, §7; issue #38, backlog 0012).

Fixtures follow the same v2 DataItem-construction pattern as
tests/test_coreset.py's own `_item` helper (real DataItem objects built
through derive_release_class/derive_cloud_ok, never hand-set release_class/
cloud_ok), extended with the `speaker_generation`/`diff_catalog_flags`
parameters this module's selection logic actually reads.
"""

import hashlib
import json
import unicodedata
from pathlib import Path

from data_contract import DataItem, ReleaseClassInputs, derive_cloud_ok, derive_release_class
from governance.store import manifest_sha256 as governance_manifest_sha256
from coreset import CoverageScorecard, CoverageStatus, CoverageTargets, DiffCatalogCell
from coreset.selection import (
    CoresetSelectionError,
    SelectionResult,
    manifest_sha256_for_items,
    select_coreset,
    text_source_from_sidecar_dir,
    write_run_manifest,
)


def _item(
    item_id: str,
    *,
    language_tag: str = "frc",
    eligible: bool = True,
    speaker_id: str | None = "spk-lafourche-1",
    lect: str | None = "Lafourche",
    genre: str = "conversation",
    speaker_generation: str = "elder_fluent",
    diff_catalog_flags: list[str] | None = None,
) -> DataItem:
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
        diff_catalog_flags=diff_catalog_flags or [],
        cloud_ok=cloud_ok,
        schema_version="2.0.0",
    )


def _scorecard(
    *,
    gate_class: dict[str, int] | None = None,
    diff_catalog_coverage: dict[str, CoverageStatus] | None = None,
) -> CoverageScorecard:
    return CoverageScorecard(
        language="frc",
        schema_version="2.0.0",
        targets=CoverageTargets(floor={"items": 5, "speakers": 3}, aspirational={"items": 20, "speakers": 5}),
        observed_counts={"items": 0, "speakers": 0},
        floor_verdicts={},
        aspirational_verdicts={},
        diff_catalog_coverage=diff_catalog_coverage or {},
        gate_class=gate_class or {},
        base_failure_rate={},
        unmet_cells=[cell_id for cell_id, status in (diff_catalog_coverage or {}).items() if status == "unmet"],
        next_collection_priorities=[],
        annotation_hours_committed=0.0,
        annotation_hours_budget_note="n/a",
        stratified_counts={},
    )


# --- eligibility pre-filter -------------------------------------------------


def test_selection_runs_eligibility_first_and_excludes_ineligible() -> None:
    items = [
        _item("eligible-1", eligible=True),
        _item("eligible-2", eligible=True, speaker_id="spk-2"),
        _item("ineligible-1", eligible=False),
    ]
    result = select_coreset(items, k=10, seed=1)
    assert "ineligible-1" not in result.selected_ids
    assert set(result.selected_ids) == {"eligible-1", "eligible-2"}


# --- stratification / speaker_generation floor ------------------------------


def test_stratification_respects_speaker_generation_floor() -> None:
    """A small elder_fluent group must not be rounded to zero purely by
    proportional allocation against a much larger heritage group."""
    items = []
    for i in range(2):
        items.append(
            _item(
                f"elder-{i}",
                speaker_id=f"spk-elder-{i}",
                speaker_generation="elder_fluent",
                lect="Lafourche",
            )
        )
    for i in range(20):
        items.append(
            _item(
                f"heritage-{i}",
                speaker_id=f"spk-heritage-{i}",
                speaker_generation="heritage",
                lect="Terrebonne",
            )
        )

    result = select_coreset(items, k=6, seed=7)

    selected_by_id = {item.item_id: item for item in items if item.item_id in result.selected_ids}
    elder_selected = [i for i in selected_by_id if i.startswith("elder-")]
    assert len(elder_selected) >= 1, "elder_fluent floor should guarantee at least one pick"


def test_quotas_fill_to_k_when_pool_allows() -> None:
    """Regression: six eligible items, each its own singleton stratum (six
    distinct lects) — the naive proportional share alone rounds every
    stratum's quota to zero at k=3 (share=0.5 each, int(round(0.5))=0), and
    with no speaker_generation floor group small enough to rescue it the
    two-pass quota calculation used to under-fill: k=3 returned only 2
    items, and without any floor at all it returned 0. The largest-remainder
    fill pass must top the total back up to min(k, pool size)."""
    items = [_item(f"item-{i}", speaker_id=f"spk-{i}", lect=f"lect-{i}") for i in range(6)]

    result_k3 = select_coreset(items, k=3, seed=1)
    assert len(result_k3.selected_ids) == 3

    result_k10 = select_coreset(items, k=10, seed=1)
    assert len(result_k10.selected_ids) == 6  # never more than the eligible pool


# --- greedy k-center diversity ----------------------------------------------


def test_k_center_selects_diverse_items_on_synthetic_clusters() -> None:
    """Two well-separated clusters (distinguished by genre, part of the
    categorical feature vector) — a k=2 selection should pick one item from
    each cluster rather than two items from the same cluster."""
    cluster_a = [
        _item(f"a-{i}", speaker_id=f"spk-a-{i}", genre="conversation", lect="Lafourche")
        for i in range(5)
    ]
    cluster_b = [
        _item(f"b-{i}", speaker_id=f"spk-b-{i}", genre="song_performance", lect="Lafourche")
        for i in range(5)
    ]
    items = cluster_a + cluster_b

    result = select_coreset(items, k=2, seed=3, stratify=())

    picked_a = any(item_id.startswith("a-") for item_id in result.selected_ids)
    picked_b = any(item_id.startswith("b-") for item_id in result.selected_ids)
    assert picked_a and picked_b, f"expected one pick from each cluster, got {result.selected_ids}"


# --- priority boost for unmet cells -----------------------------------------


def test_priority_boost_prefers_unmet_cell_items() -> None:
    """With lambda_relevance weighted toward relevance, an item flagged
    against an unmet diff-catalog cell should be preferred over an
    otherwise-identical item with no flags, even before diversity considerations."""
    scorecard = _scorecard(
        gate_class={"VERB-001": 2},
        diff_catalog_coverage={"VERB-001": "unmet"},
    )
    flagged = [
        _item(f"flagged-{i}", speaker_id=f"spk-f-{i}", diff_catalog_flags=["VERB-001"])
        for i in range(3)
    ]
    unflagged = [_item(f"plain-{i}", speaker_id=f"spk-p-{i}") for i in range(3)]
    items = flagged + unflagged

    result = select_coreset(
        items, k=1, seed=1, stratify=(), scorecard=scorecard, lambda_relevance=0.9
    )
    assert result.selected_ids[0].startswith("flagged-")


# --- determinism -------------------------------------------------------------


def test_selection_is_deterministic_for_seed() -> None:
    items = [
        _item(f"item-{i}", speaker_id=f"spk-{i}", genre="conversation" if i % 2 == 0 else "interview")
        for i in range(12)
    ]
    result_1 = select_coreset(items, k=5, seed=42)
    result_2 = select_coreset(items, k=5, seed=42)
    assert result_1.selected_ids == result_2.selected_ids
    assert result_1.per_stratum_counts == result_2.per_stratum_counts


# --- manifest ----------------------------------------------------------------


def test_manifest_json_shape_and_pinned_sha256(tmp_path: Path) -> None:
    items = [_item("item-b"), _item("item-a", speaker_id="spk-2")]
    manifest_path = tmp_path / "manifest_run-1.json"
    write_run_manifest(manifest_path, "run-1", items)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"
    assert data["schema_version"] == "2.0.0"
    assert data["items"] == [
        {"item_id": "item-a", "sha256": "item-a"},
        {"item_id": "item-b", "sha256": "item-b"},
    ]
    # Pinned fixture value (issue #38 wave preamble rule 7: "Where your
    # ticket needs the manifest hash, pin its value on a fixture in a
    # test"): sha256 hex of the UTF-8 bytes of
    # '[{"item_id":"item-a","sha256":"item-a"},{"item_id":"item-b","sha256":"item-b"}]'
    # (the shared format's exact json.dumps(items, sort_keys=True,
    # separators=(",", ":")) form), computed independently and pinned as a
    # literal so a future accidental change to the hash formula is caught
    # even if this test's own payload-construction logic changed to match.
    pinned_sha256 = "0253a3e493cc5df6afaa8ee86c89c8016c14c4af7935e0aba2cb4613d84625d0"
    assert data["manifest_sha256"] == pinned_sha256
    assert manifest_sha256_for_items(items) == pinned_sha256


def test_manifest_sha256_ignores_item_order() -> None:
    a = _item("item-a")
    b = _item("item-b", speaker_id="spk-2")
    assert manifest_sha256_for_items([a, b]) == manifest_sha256_for_items([b, a])


# --- coverage_gain_by_cell ----------------------------------------------------


def test_coverage_gain_reported_per_cell() -> None:
    scorecard = _scorecard(
        gate_class={"VERB-001": 1},
        diff_catalog_coverage={"VERB-001": "unmet"},
    )
    items = [
        _item(f"item-{i}", speaker_id=f"spk-{i}", diff_catalog_flags=["VERB-001"])
        for i in range(3)
    ]
    result = select_coreset(items, k=2, seed=1, stratify=(), scorecard=scorecard)
    assert result.coverage_gain_by_cell.get("VERB-001") == len(result.selected_ids)


# --- input validation ---------------------------------------------------------


def test_select_coreset_rejects_non_positive_k() -> None:
    items = [_item("a")]
    try:
        select_coreset(items, k=0, seed=1)
    except CoresetSelectionError:
        pass
    else:
        raise AssertionError("expected CoresetSelectionError for k=0")


def test_select_coreset_rejects_unknown_stratify_field() -> None:
    items = [_item("a")]
    try:
        select_coreset(items, k=1, seed=1, stratify=("register",))
    except CoresetSelectionError:
        pass
    else:
        raise AssertionError("expected CoresetSelectionError for an unsupported stratify field")


def test_selection_result_is_a_dataclass_with_expected_fields() -> None:
    result = SelectionResult(selected_ids=("a",), per_stratum_counts={"x": 1}, coverage_gain_by_cell={})
    assert result.selected_ids == ("a",)
    assert result.text_missing_ids == ()


# --- text-in seam (issue #45) --------------------------------------------


def test_text_source_changes_ngram_bag_vs_item_id_proxy() -> None:
    """Two items with the same item_id-derived proxy signal but genuinely
    different real text must select differently once a text_source is
    supplied — proof the n-gram bag is actually hashing the real text, not
    silently still hashing item_id underneath."""
    item_a = _item("item-a", speaker_id="spk-a", genre="conversation", lect="Lafourche")
    item_b = _item("item-b", speaker_id="spk-b", genre="conversation", lect="Lafourche")
    items = [item_a, item_b]

    texts = {
        "item-a": "Mo té we li nan gran sitiyasyon ki te difisil anpil pou tout moun.",
        "item-b": "Zong nof kzzq lmxx wtrb hjkq vfpo yudl bnmc zxqw asdr ghjk.",
    }

    def text_source(item: DataItem) -> str | None:
        return texts.get(item.item_id)

    result_with_text = select_coreset(items, k=1, seed=1, stratify=(), text_source=text_source)
    result_proxy = select_coreset(items, k=1, seed=1, stratify=())

    # Both selections must be an actual, valid pick (k=1 from 2 candidates);
    # the two runs need not disagree in general, but the two feature
    # vectors themselves must differ so this isn't a no-op wiring — checked
    # directly via the module's own feature-vector builder rather than only
    # via the (looser) selected_ids outcome.
    from coreset.selection import DEFAULT_NGRAM_DIMENSION, _feature_vector

    features_text_a = _feature_vector(
        item_a, scorecard=None, ngram_dimension=DEFAULT_NGRAM_DIMENSION, text_source=text_source
    )
    features_proxy_a = _feature_vector(
        item_a, scorecard=None, ngram_dimension=DEFAULT_NGRAM_DIMENSION, text_source=None
    )
    assert features_text_a != features_proxy_a
    assert result_with_text.selected_ids[0] in {"item-a", "item-b"}
    assert result_proxy.selected_ids[0] in {"item-a", "item-b"}


def test_text_source_none_falls_back_to_proxy_and_reports_missing_ids() -> None:
    have_text = _item("has-text", speaker_id="spk-1")
    no_text = _item("no-text", speaker_id="spk-2")
    items = [have_text, no_text]

    def text_source(item: DataItem) -> str | None:
        return "some real sentence here" if item.item_id == "has-text" else None

    result = select_coreset(items, k=2, seed=1, stratify=(), text_source=text_source)
    assert result.text_missing_ids == ("no-text",)

    # No text_source at all: nothing is "missing" (the proxy is simply used
    # for everyone, matching every pre-#45 caller unchanged).
    result_no_source = select_coreset(items, k=2, seed=1, stratify=())
    assert result_no_source.text_missing_ids == ()


def test_manifest_item_sha256_is_nfc_text_hash_when_text_supplied(tmp_path: Path) -> None:
    item = _item("item-a", speaker_id="spk-2")
    raw_text = "Café, naïve, coöperate"  # deliberately containing combining/precomposed forms
    nfc_text = unicodedata.normalize("NFC", raw_text)
    expected_sha256 = hashlib.sha256(nfc_text.encode("utf-8")).hexdigest()

    def text_source(candidate: DataItem) -> str | None:
        return raw_text if candidate.item_id == "item-a" else None

    manifest_path = tmp_path / "manifest_run-text.json"
    write_run_manifest(manifest_path, "run-text", [item], text_source=text_source)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["items"] == [{"item_id": "item-a", "sha256": expected_sha256}]
    assert expected_sha256 != item.item_id


def test_manifest_sha256_formula_unchanged_pinned_fixture(tmp_path: Path) -> None:
    """The exact #38 pinned-fixture regression, now also run explicitly with
    `text_source=None` to pin that the outer formula is untouched by #45."""
    items = [_item("item-b"), _item("item-a", speaker_id="spk-2")]
    manifest_path = tmp_path / "manifest_run-1.json"
    write_run_manifest(manifest_path, "run-1", items, text_source=None)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["items"] == [
        {"item_id": "item-a", "sha256": "item-a"},
        {"item_id": "item-b", "sha256": "item-b"},
    ]
    pinned_sha256 = "0253a3e493cc5df6afaa8ee86c89c8016c14c4af7935e0aba2cb4613d84625d0"
    assert data["manifest_sha256"] == pinned_sha256
    assert manifest_sha256_for_items(items, text_source=None) == pinned_sha256
    assert manifest_sha256_for_items(items) == pinned_sha256


def test_governance_auditor_agrees_on_text_hashed_manifest(tmp_path: Path) -> None:
    item_a = _item("item-a", speaker_id="spk-2")
    item_b = _item("item-b")
    items = [item_a, item_b]

    def text_source(candidate: DataItem) -> str | None:
        return {"item-a": "premye fraz la", "item-b": "dezyèm fraz la"}[candidate.item_id]

    manifest_path = tmp_path / "manifest_run-audit.json"
    write_run_manifest(manifest_path, "run-audit", items, text_source=text_source)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    recomputed = governance_manifest_sha256(manifest_path)
    assert recomputed == data["manifest_sha256"]
    assert recomputed == manifest_sha256_for_items(items, text_source=text_source)


def test_selection_deterministic_under_text_source() -> None:
    items = [
        _item(f"item-{i}", speaker_id=f"spk-{i}", genre="conversation" if i % 2 == 0 else "interview")
        for i in range(12)
    ]
    texts = {item.item_id: f"sentence number {i} with some real words in it" for i, item in enumerate(items)}

    def text_source(item: DataItem) -> str | None:
        return texts[item.item_id]

    result_1 = select_coreset(items, k=5, seed=42, text_source=text_source)
    result_2 = select_coreset(items, k=5, seed=42, text_source=text_source)
    assert result_1.selected_ids == result_2.selected_ids
    assert result_1.per_stratum_counts == result_2.per_stratum_counts


def test_text_source_from_sidecar_dir_reads_layout(tmp_path: Path) -> None:
    raw_text = "Yon fraz egzanp pou tès la."
    nfc_text = unicodedata.normalize("NFC", raw_text)
    item_id = hashlib.sha256(nfc_text.encode("utf-8")).hexdigest()

    (tmp_path / "some-file.txt").write_text(raw_text, encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"filename": "some-file.txt", "source": "test-collection"}) + "\n",
        encoding="utf-8",
    )

    text_source = text_source_from_sidecar_dir(tmp_path)
    item = _item(item_id, speaker_id="spk-sidecar")
    assert text_source(item) == nfc_text


def test_text_source_from_sidecar_dir_missing_file_is_none(tmp_path: Path) -> None:
    # An item_id that matches no file under raw_dir at all.
    text_source = text_source_from_sidecar_dir(tmp_path)
    item = _item("no-such-content-hash", speaker_id="spk-x")
    assert text_source(item) is None

    # raw_dir itself does not exist.
    missing_dir_source = text_source_from_sidecar_dir(tmp_path / "does-not-exist")
    assert missing_dir_source(item) is None
