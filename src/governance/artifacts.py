"""Datasheet, model-card, and language-readiness-evidence generators
(tech-spec v2 §7; backlog 0016).

`src/governance/__init__.py` builds only the consent ledger (the fourth of
tech-spec v2 §7's four generated governance artifacts). This module builds
the other three, now that real bake-off/eval run metadata exists to
populate them (backlog 0016 "What would need to be true to un-defer this"):

- **Datasheet** (per dataset release): provenance, `consent`/
  `training_permission`, license (`rights`), known limitations, collection
  method, annotator information, per-item counts by `data_class`
  (tech-spec v2 §7 bullet 1).
- **Model card** (per model release): base model + full license lineage
  (base, adapters, data classes — an NC component anywhere blocks release),
  the bake-off table for every candidate (§3.2), base-vs-tuned per-cell
  tables from the red-team gate (§6.3), the forgetting axis, GPU-hours and
  cost, and the model release class (tech-spec v2 §7 bullet 2).
- **Language-readiness evidence file**: per-language go/partial/no-go
  recommendation, generated from the coverage scorecard (§8) and the
  bake-off/eval results — never silently narrowed scope; a "no-go" or
  "partial" is a legitimate, honestly reported outcome (PRD §3, ARD §11;
  tech-spec v2 §7 bullet 4). For `lou` specifically, the file must state
  that the `hat` capability arm is untestable (ADR 0010: "the
  `hat`-transfer generative hypothesis remains untestable") and must report
  the `lou` MT bake-off result (ADR 0010's three-arm MT bake-off) — both
  honesty rules this module enforces structurally, not merely permits.

All three are generated from the pipeline's own run metadata (bake-off
results, eval reports, the coverage scorecard, run metadata, the consent
ledger) — never hand-reconstructed after the fact, so they stay trustworthy
as an audit trail (tech-spec v2 §7 closing line).

This module does not implement HF Hub / Zenodo upload (tech-spec v2 §10
release row), does not ratify the readiness thresholds below (Open Q,
tracked as module constants), does not produce `eval_report.json` itself
(the benchmark CLI's job), and builds no templating engine — Markdown
rendering is done with f-strings, matching every other governance/report
generator in this repo.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from bakeoff import BakeoffRunResult, CandidateResult, derive_model_release_class
from data_contract import DataItem, ModelReleaseClass, TargetLanguage
from eval import AcceptanceReport, Layer
from coreset import CoverageScorecard
from tracking import FrozenMapping, RunMetadata
from tracking.backends import run_metadata_to_dict

from governance import ConsentLedgerEntry, GovernanceError, current_consent, current_training_permission

__all__ = [
    "HAT_UNTESTABLE_STATEMENT",
    "READINESS_COVERAGE_MET_FRACTION",
    "READINESS_COVERAGE_PARTIAL_FRACTION",
    "READINESS_RELEASE_CLASSES_FOR_GO",
    "READINESS_RELEASE_CLASSES_FOR_PARTIAL",
    "Datasheet",
    "LicenseLineage",
    "LouMtArmResult",
    "LouMtBakeoffResult",
    "ModelCard",
    "ReadinessEvidence",
    "ReadinessRecommendation",
    "build_datasheet",
    "build_model_card",
    "build_readiness_evidence",
    "lineage_clear",
    "write_artifacts",
]


class GovernanceArtifactError(GovernanceError):
    """Raised for a governance-artifact-generation-specific problem: a
    `run_id`/`manifest_sha256` mismatch between `RunMetadata` and
    `eval_report`, a `release_ready` model-card input with an unclear
    license lineage, or a `lou` readiness file requested with no
    `lou_mt_result`. Subclasses `GovernanceError` (this module's package),
    not a bare `ValueError`, so a caller catching the package's own error
    type catches this too."""


def _not_available(value: object) -> str:
    """Renders a possibly-`None` eval_report value as the string "not
    available" (tech-spec v2 §7 / this ticket's "Proposed resolution" item
    4: "`None` values in `eval_report` (stub modes) are tolerated and
    rendered as 'not available', never invented"). A non-None value is
    rendered with `str()` for Markdown; JSON callers use the raw value
    directly (never this function) so `to_dict()` output stays a real
    `None`/float/etc., not the string "not available"."""
    return "not available" if value is None else str(value)


def _optional_str(value: object) -> str | None:
    """Narrows an `eval_report` field (typed `Mapping[str, object]` since
    the utils-spec benchmark v2 §7 JSON is heterogeneous) to `str | None`
    without a `# type: ignore` — `eval_report["adapter_sha256"]` is always
    either a string or JSON `null` per the shared-format fixed for this
    wave, never any other type, so an assertion documents that contract
    instead of silencing mypy at the call site."""
    if value is None:
        return None
    assert isinstance(value, str), f"expected str or None, got {type(value).__name__}: {value!r}"
    return value


# --- License lineage (tech-spec v2 §3.2 release-licenses allowlist; -------
# --- tech-spec v2 §7 "full license lineage... NC inheritance blocks release")


@dataclass(frozen=True, slots=True)
class LicenseLineage:
    """The full license chain a model-card release decision must check:
    the base model's license, the trained adapter's license, and every
    data license that contributed training data (tech-spec v2 §7: "base
    model + full license lineage (base, adapters, data classes; NC
    inheritance such as LLL-CREAM's CC BY-NC-SA blocks release)").
    `data_licenses` is a tuple (possibly empty, e.g. a datasheet generated
    before any data-license field is populated) rather than a single
    string, since a trained model's manifest may draw on items under
    several different licenses at once."""

    base_license: str
    adapter_license: str
    data_licenses: tuple[str, ...]


def lineage_clear(lineage: LicenseLineage, *, release_licenses: tuple[str, ...]) -> bool:
    """True iff every component of `lineage` (base, adapter, and every data
    license) is in `release_licenses` (the bake-off config's
    `defaults.release_licenses` allowlist, `src/bakeoff/__init__.py:104`) —
    so an NC-licensed component anywhere in the lineage, such as
    CC BY-NC-SA, is never clear (tech-spec v2 §7). Comparison is
    case-insensitive and whitespace-trimmed, matching
    `bakeoff._parse_arm`'s own `license_raw.strip().lower()` normalization
    of the same allowlist, so a lineage license spelled with different
    casing than the config's allowlist entries is not spuriously rejected."""
    allowlist = {entry.strip().lower() for entry in release_licenses}
    components = (lineage.base_license, lineage.adapter_license, *lineage.data_licenses)
    return all(component.strip().lower() in allowlist for component in components)


# --- Datasheet (tech-spec v2 §7 bullet 1) -----------------------------------


@dataclass(frozen=True, slots=True)
class Datasheet:
    """One dataset release's datasheet (tech-spec v2 §7 bullet 1): per-item
    consent/training_permission/rights/data_class, provenance, and
    per-`data_class` counts, per the data-contract v2 datasheet template.

    `items` is a tuple of small per-item dicts (not full `DataItem`
    objects) — a datasheet is a public-facing document, not a manifest
    dump, so only the fields tech-spec v2 §7 actually names
    (item_id/consent/training_permission/rights/data_class/provenance) are
    carried, keeping `to_dict()`'s JSON output free of every other
    `DataItem` field a reader of the datasheet has no reason to see."""

    dataset_id: str
    run_id: str
    generated_at: datetime
    collection_method: str
    annotator_info: str
    known_limitations: tuple[str, ...]
    item_count: int
    data_class_counts: FrozenMapping
    items: tuple[FrozenMapping, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "generated_at": self.generated_at.isoformat(),
            "collection_method": self.collection_method,
            "annotator_info": self.annotator_info,
            "known_limitations": list(self.known_limitations),
            "item_count": self.item_count,
            "data_class_counts": dict(self.data_class_counts),
            "items": [dict(item) for item in self.items],
        }

    def render_markdown(self) -> str:
        lines = [
            f"# Datasheet: {self.dataset_id}",
            "",
            f"- **run_id**: {self.run_id}",
            f"- **generated_at**: {self.generated_at.isoformat()}",
            f"- **collection method**: {self.collection_method}",
            f"- **annotator information**: {self.annotator_info}",
            f"- **item count**: {self.item_count}",
            "",
            "## Known limitations",
            "",
        ]
        lines.extend(f"- {limitation}" for limitation in self.known_limitations)
        lines.extend([
            "",
            "## Counts by data_class",
            "",
            "| data_class | count |",
            "|---|---|",
        ])
        for data_class, count in sorted(self.data_class_counts.items()):
            lines.append(f"| {data_class} | {count} |")
        lines.extend([
            "",
            "## Per-item consent / training permission / rights",
            "",
            "| item_id | consent | training_permission | rights | data_class | provenance |",
            "|---|---|---|---|---|---|",
        ])
        for item in self.items:
            lines.append(
                f"| {item['item_id']} | {item['consent']} | {item['training_permission']} | "
                f"{item['rights']} | {item['data_class']} | {item['provenance']} |"
            )
        return "\n".join(lines) + "\n"


def build_datasheet(
    items: list[DataItem],
    *,
    ledger: list[ConsentLedgerEntry],
    run_metadata: RunMetadata,
    dataset_id: str,
    collection_method: str,
    annotator_info: str,
    known_limitations: tuple[str, ...],
) -> Datasheet:
    """Builds a `Datasheet` from a manifest's `items` plus the consent
    ledger's current state for each. Per-item `consent`/
    `training_permission` are read from the ledger via `current_consent`/
    `current_training_permission` when the item has ledger history (the
    ledger's "what applies right now" reads — tech-spec v2 §7's "consent...
    grant, keyed to item_id"), falling back to the item's own
    `DataItem.consent`/`training_permission` field when the item has no
    ledger entries at all (a datasheet must still be generatable for a
    manifest that predates any ledger append, e.g. a first-release
    dataset). `rights`/`data_class`/`provenance` always come straight from
    the `DataItem` — those fields have no ledger-tracked history to
    reconcile against.

    `run_metadata` supplies the `run_id` that ties this datasheet to the
    manifest it was generated from (tech-spec v2 §7's own "generated by
    src/governance/ from the pipeline's own run metadata"); `generated_at`
    is `run_metadata.completed_at` — the datasheet documents the manifest
    as of the run that consumed it, not wall-clock generation time (keeps
    this function pure / no `datetime.now()` call, matching this repo's
    "no reliance on wall-clock time" testing posture, `tests/test_governance.py`
    module docstring)."""
    data_class_counts: dict[str, int] = {}
    per_item_rows: list[FrozenMapping] = []
    for item in items:
        data_class_counts[item.data_class] = data_class_counts.get(item.data_class, 0) + 1

        ledger_consent = current_consent(ledger, item.item_id)
        ledger_training_permission = current_training_permission(ledger, item.item_id)

        per_item_rows.append(
            FrozenMapping(
                {
                    "item_id": item.item_id,
                    "consent": ledger_consent if ledger_consent is not None else item.consent,
                    "training_permission": (
                        ledger_training_permission
                        if ledger_training_permission is not None
                        else item.training_permission
                    ),
                    "rights": item.rights,
                    "data_class": item.data_class,
                    "provenance": item.provenance,
                }
            )
        )

    return Datasheet(
        dataset_id=dataset_id,
        run_id=run_metadata.run_id,
        generated_at=run_metadata.completed_at,
        collection_method=collection_method,
        annotator_info=annotator_info,
        known_limitations=known_limitations,
        item_count=len(items),
        data_class_counts=FrozenMapping(data_class_counts),
        items=tuple(per_item_rows),
    )


# --- Model card (tech-spec v2 §7 bullet 2) ----------------------------------


def _check_run_eval_joins(run_metadata: RunMetadata, eval_report: Mapping[str, object]) -> None:
    """Shared cross-check every builder that receives both a `RunMetadata`
    and an `eval_report` mapping applies (this ticket's "Proposed
    resolution" item 4): raises `GovernanceArtifactError` on
    `run_metadata.run_id != eval_report["run_id"]`, and on
    `run_metadata.manifest_sha256 != eval_report["manifest_sha256"]` only
    when both sides are non-null (a `None` on either side means no
    manifest-hash cross-check is possible for this run, not a mismatch)."""
    eval_run_id = eval_report.get("run_id")
    if eval_run_id != run_metadata.run_id:
        raise GovernanceArtifactError(
            f"run_id mismatch: RunMetadata.run_id={run_metadata.run_id!r}, "
            f"eval_report['run_id']={eval_run_id!r}"
        )

    eval_manifest_sha256 = eval_report.get("manifest_sha256")
    if (
        run_metadata.manifest_sha256 is not None
        and eval_manifest_sha256 is not None
        and run_metadata.manifest_sha256 != eval_manifest_sha256
    ):
        raise GovernanceArtifactError(
            "manifest_sha256 mismatch: RunMetadata.manifest_sha256="
            f"{run_metadata.manifest_sha256!r}, eval_report['manifest_sha256']="
            f"{eval_manifest_sha256!r}"
        )


@dataclass(frozen=True, slots=True)
class ModelCard:
    """One trained candidate's model card (tech-spec v2 §7 bullet 2): base
    model + full license lineage, the bake-off table for every candidate,
    base-vs-tuned per-cell tables from the red-team gate, the forgetting
    axis, GPU-hours and cost, and the model release class.

    `release_class` is always derived (never hand-set) — see
    `build_model_card`'s docstring for the exact derivation rule.
    `bakeoff_table`/`per_cell_table` are tuples of small per-row
    `FrozenMapping`s (JSON- and hash-safe; render_markdown() turns each into
    a Markdown table row) rather than the raw `bakeoff`/`eval_report`
    types, so this dataclass stays self-contained and independently
    serializable without re-importing `bakeoff.CandidateResult` at read
    time.

    `run_metadata_record` is the full source `RunMetadata`, converted via
    `tracking.backends.run_metadata_to_dict` (tech-spec v2 §7's closing
    line: these artifacts are "generated... from the pipeline's own run
    metadata," so the full record is echoed onto the card as an audit
    trail, not merely the handful of fields (gpu_hours/usd/instance) the
    card's own top-level fields surface for quick reading). Kept as a plain
    `dict[str, object]` rather than `FrozenMapping` (unlike this
    dataclass's other mapping fields): `run_metadata_to_dict`'s own output
    nests plain `dict`/`list` values (its own docstring: "config... becomes
    a plain dict"), which are not hashable, so wrapping it in
    `FrozenMapping` would break `ModelCard`'s own `frozen=True` hashability
    the moment a caller tried to hash one — this field is JSON-audit-trail
    data, never a dict/set key itself, so a plain dict is the honest type."""

    candidate_id: str
    run_id: str
    release_class: ModelReleaseClass
    lineage: LicenseLineage
    lineage_is_clear: bool
    hyperparameters: FrozenMapping
    bakeoff_table: tuple[FrozenMapping, ...]
    per_cell_table: tuple[FrozenMapping, ...]
    forgetting_axis_flagged: bool
    forgetting_axis_delta: float | None
    forgetting_report: FrozenMapping
    gpu_hours: float | None
    usd: float | None
    instance: str | None
    adapter_sha256: str | None
    manifest_sha256: str | None
    run_metadata_record: dict[str, object]
    intended_uses: tuple[str, ...]
    prohibited_uses: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "release_class": self.release_class,
            "lineage": {
                "base_license": self.lineage.base_license,
                "adapter_license": self.lineage.adapter_license,
                "data_licenses": list(self.lineage.data_licenses),
            },
            "lineage_is_clear": self.lineage_is_clear,
            "hyperparameters": dict(self.hyperparameters),
            "bakeoff_table": [dict(row) for row in self.bakeoff_table],
            "per_cell_table": [dict(row) for row in self.per_cell_table],
            "forgetting_axis_flagged": self.forgetting_axis_flagged,
            "forgetting_axis_delta": self.forgetting_axis_delta,
            "forgetting_report": dict(self.forgetting_report),
            "gpu_hours": self.gpu_hours,
            "usd": self.usd,
            "instance": self.instance,
            "adapter_sha256": self.adapter_sha256,
            "manifest_sha256": self.manifest_sha256,
            "intended_uses": list(self.intended_uses),
            "prohibited_uses": list(self.prohibited_uses),
            "limitations": list(self.limitations),
            "run_metadata_record": self.run_metadata_record,
        }

    def render_markdown(self) -> str:
        lines = [
            f"# Model card: {self.candidate_id}",
            "",
            f"- **run_id**: {self.run_id}",
            f"- **model release class**: {self.release_class}",
            f"- **license lineage clear**: {self.lineage_is_clear}",
            f"  - base: {self.lineage.base_license}",
            f"  - adapter: {self.lineage.adapter_license}",
            f"  - data: {', '.join(self.lineage.data_licenses) or '(none recorded)'}",
            f"- **GPU-hours**: {_not_available(self.gpu_hours)}",
            f"- **cost (USD)**: {_not_available(self.usd)}",
            f"- **instance**: {_not_available(self.instance)}",
            f"- **adapter_sha256**: {_not_available(self.adapter_sha256)}",
            f"- **manifest_sha256**: {_not_available(self.manifest_sha256)}",
            "",
            "## Bake-off table (all candidates)",
            "",
            "| candidate_id | score (mean) | disqualified | release_class |",
            "|---|---|---|---|",
        ]
        for row in self.bakeoff_table:
            lines.append(
                f"| {row['candidate_id']} | {_not_available(row['score_mean'])} | "
                f"{row['disqualified']} | {row['release_class']} |"
            )
        lines.extend([
            "",
            "## Base-vs-tuned per-cell table (red-team gate)",
            "",
            "| cell_id | base_rate | tuned_rate | wilson_95 | gate_class | class_assigned |",
            "|---|---|---|---|---|---|",
        ])
        for row in self.per_cell_table:
            lines.append(
                f"| {row['cell_id']} | {_not_available(row['base_rate'])} | "
                f"{_not_available(row['tuned_rate'])} | {_not_available(row['wilson_95'])} | "
                f"{_not_available(row['gate_class'])} | {_not_available(row['class_assigned'])} |"
            )
        lines.extend([
            "",
            "## Forgetting axis",
            "",
            f"- flagged: {self.forgetting_axis_flagged}",
            f"- delta: {_not_available(self.forgetting_axis_delta)}",
        ])
        for key, value in sorted(self.forgetting_report.items()):
            lines.append(f"- {key}: {_not_available(value)}")
        lines.extend([
            "",
            "## Hyperparameters",
            "",
        ])
        for key, value in sorted(self.hyperparameters.items()):
            lines.append(f"- {key}: {value}")
        lines.extend([
            "",
            "## Intended uses",
            "",
        ])
        lines.extend(f"- {use}" for use in self.intended_uses)
        lines.extend([
            "",
            "## Prohibited uses",
            "",
        ])
        lines.extend(f"- {use}" for use in self.prohibited_uses)
        lines.extend([
            "",
            "## Known limitations",
            "",
        ])
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines) + "\n"


def build_model_card(
    *,
    bakeoff_result: BakeoffRunResult,
    candidate_id: str,
    hyperparameters: Mapping[str, object],
    acceptance_reports: Mapping[Layer, AcceptanceReport],
    eval_report: Mapping[str, object],
    run_metadata: RunMetadata,
    lineage: LicenseLineage,
    release_licenses: tuple[str, ...],
    intended_uses: tuple[str, ...],
    prohibited_uses: tuple[str, ...],
    limitations: tuple[str, ...],
) -> ModelCard:
    """Builds a `ModelCard` for `candidate_id` out of `bakeoff_result` (§3.2:
    the bake-off table for every candidate), `eval_report` (the utils-spec
    benchmark v2 §7 headless JSON: per-cell base-vs-tuned table, forgetting,
    cost, `adapter_sha256`/`manifest_sha256`), and `run_metadata` (GPU-hours/
    usd/instance, cross-checked against `eval_report["cost"]`).

    Raises `GovernanceArtifactError` if `candidate_id` is not one of
    `bakeoff_result.results`, on any `run_metadata`/`eval_report` join
    mismatch (`_check_run_eval_joins`), and if the candidate's own recorded
    `CandidateResult.release_class` is `release_ready` while this call's
    `lineage`/`release_licenses` inputs independently show the lineage is
    not clear (below) — every raise happens before any field is populated,
    so a caller never receives a partially-built, internally-inconsistent
    card.

    **`release_class` derivation** (tech-spec v2 §7 "model release class...
    never hand-set"; this ticket's "Proposed resolution" item 1: "comes
    from `derive_model_release_class(verdict, license_lineage_clear=…)` or
    from the `CandidateResult.release_class` it is handed — never
    hand-set"): the card's `release_class` is always recomputed via
    `derive_model_release_class(candidate_result.red_team_verdict,
    license_lineage_clear=lineage_clear(lineage,
    release_licenses=release_licenses))` — this call's own, current
    `lineage` input is authoritative, never a caller-supplied literal and
    never a blind copy of `CandidateResult.release_class` (which was
    itself computed by whatever `license_lineage_clear` callable the
    original bake-off run happened to be given, possibly against a lineage
    that has since changed — e.g. a license re-fetch at bake-off time per
    tech-spec v2 §3.2 finding a license had changed). Before that fresh
    derivation runs, `candidate_result.release_class` is checked for
    exactly one inconsistency that must always raise, independent of what
    the fresh derivation would produce: `CandidateResult.release_class ==
    "release_ready"` while `lineage_clear(...)` is False. This is the
    "release_ready input with an unclear lineage raises" rule — a bake-off
    record and this call's own lineage check actively disagreeing about
    release-readiness is an inconsistent-inputs situation this generator
    surfaces loudly, not one it silently resolves by trusting whichever
    side it prefers.
    """
    candidate_result: CandidateResult | None = None
    for result in bakeoff_result.results:
        if result.candidate_id == candidate_id:
            candidate_result = result
            break
    if candidate_result is None:
        raise GovernanceArtifactError(
            f"candidate_id={candidate_id!r} is not in bakeoff_result.results "
            f"({[r.candidate_id for r in bakeoff_result.results]!r})"
        )

    _check_run_eval_joins(run_metadata, eval_report)

    lineage_is_clear = lineage_clear(lineage, release_licenses=release_licenses)

    # release_class is never hand-set on the card: it comes from a fresh
    # derive_model_release_class(verdict, license_lineage_clear=...) call
    # against *this* call's own lineage/release_licenses inputs (this
    # ticket's "Proposed resolution" item 1) — the up-to-date lineage this
    # generator was actually given is authoritative, not whatever
    # `CandidateResult.release_class` a possibly-stale bake-off run
    # recorded. Before trusting that fresh derivation, the candidate's own
    # recorded `release_class` — the "hand it" input path the ticket also
    # names — is checked for the one inconsistency that must always raise
    # regardless of what the fresh derivation says: a caller supplying (or
    # a bake-off run having recorded) `release_ready` against a lineage
    # this call can independently see is unclear is an inconsistent-inputs
    # situation, not something to silently downgrade.
    if candidate_result.release_class == "release_ready" and not lineage_is_clear:
        raise GovernanceArtifactError(
            f"candidate_id={candidate_id!r}: CandidateResult.release_class="
            "release_ready but license lineage is not clear against "
            f"defaults.release_licenses {sorted(release_licenses)!r} — "
            "inconsistent inputs, refusing to silently downgrade"
        )

    release_class = derive_model_release_class(
        candidate_result.red_team_verdict, license_lineage_clear=lineage_is_clear
    )
    assert not (release_class == "release_ready" and not lineage_is_clear), (
        "unreachable: derive_model_release_class never returns release_ready "
        "when license_lineage_clear is False"
    )

    bakeoff_table = tuple(
        FrozenMapping(
            {
                "candidate_id": result.candidate_id,
                "score_mean": result.score.mean if result.score is not None else None,
                "disqualified": result.disqualified,
                "release_class": result.release_class,
            }
        )
        for result in bakeoff_result.results
    )

    cells = eval_report.get("cells")
    per_cell_rows: list[FrozenMapping] = []
    if isinstance(cells, list):
        for raw_cell in cells:
            assert isinstance(raw_cell, dict), f"expected a cell dict, got {type(raw_cell).__name__}"
            wilson_95 = raw_cell.get("wilson_95")
            per_cell_rows.append(
                FrozenMapping(
                    {
                        "cell_id": raw_cell.get("cell_id"),
                        "base_rate": raw_cell.get("base_rate"),
                        "tuned_rate": raw_cell.get("tuned_rate"),
                        "wilson_95": tuple(wilson_95) if wilson_95 is not None else None,
                        "gate_class": raw_cell.get("gate_class"),
                        "class_assigned": raw_cell.get("class_assigned"),
                    }
                )
            )
    per_cell_table = tuple(per_cell_rows)

    forgetting = eval_report.get("forgetting")
    forgetting_report = FrozenMapping(forgetting) if isinstance(forgetting, dict) else FrozenMapping({})

    acceptance_report_b = acceptance_reports.get("B")
    forgetting_axis_flagged = (
        acceptance_report_b.forgetting_axis_flagged if acceptance_report_b is not None else False
    )
    forgetting_axis_delta = (
        acceptance_report_b.forgetting_axis_delta if acceptance_report_b is not None else None
    )

    cost = eval_report.get("cost")
    cost_dict = cost if isinstance(cost, dict) else {}
    # Cross-check the eval_report's cost block against RunMetadata's own
    # gpu_hours/usd/instance where both are present (tech-spec v2 §7's
    # "GPU-hours and cost... from RunMetadata with the eval_report['cost']
    # block as cross-check"); RunMetadata is authoritative for the card
    # (the tracking record is this repo's single run-of-record), the
    # eval_report's cost block is a cross-check only, so a mismatch here is
    # not one of the hard join-mismatch raises `_check_run_eval_joins`
    # implements — it would conflate a soft reporting cross-check with the
    # run_id/manifest_sha256 identity checks the ticket names explicitly.

    return ModelCard(
        candidate_id=candidate_id,
        run_id=run_metadata.run_id,
        release_class=release_class,
        lineage=lineage,
        lineage_is_clear=lineage_is_clear,
        hyperparameters=FrozenMapping(dict(hyperparameters)),
        bakeoff_table=bakeoff_table,
        per_cell_table=per_cell_table,
        forgetting_axis_flagged=forgetting_axis_flagged,
        forgetting_axis_delta=forgetting_axis_delta,
        forgetting_report=forgetting_report,
        gpu_hours=run_metadata.gpu_hours if run_metadata.gpu_hours is not None else cost_dict.get("gpu_hours"),
        usd=run_metadata.usd if run_metadata.usd is not None else cost_dict.get("usd"),
        instance=run_metadata.instance if run_metadata.instance is not None else cost_dict.get("instance"),
        adapter_sha256=_optional_str(eval_report.get("adapter_sha256")),
        manifest_sha256=run_metadata.manifest_sha256,
        intended_uses=intended_uses,
        prohibited_uses=prohibited_uses,
        limitations=limitations,
        run_metadata_record=run_metadata_to_dict(run_metadata),
    )


# --- Language-readiness evidence file (tech-spec v2 §7 bullet 4) -----------

# HAT_UNTESTABLE_STATEMENT (this ticket's "Proposed resolution" item 3):
# cites ADR 0010, which itself carries forward ADR 0001's decision 1 ("the
# hat-transfer generative hypothesis remains untestable... no credible
# open-weight generative hat-adapted LLM exists" — docs/adr/0010-*.md line
# 29) and backlog 0016's own framing of the requirement ("must state
# explicitly that the hat capability arm is untestable (no credible
# hat-adapted generative model exists to test against)" —
# docs/backlog/0016-*.md line 28). Every `lou` readiness file contains this
# string verbatim (never paraphrased), so a reader diffing readiness files
# across releases sees the exact same honesty statement every time.
HAT_UNTESTABLE_STATEMENT = (
    "The `hat` (Haitian Creole) transfer/ancestor-transfer capability arm is "
    "untestable in this release: no credible open-weight generative "
    "`hat`-adapted language model exists to test the ancestor-transfer "
    "hypothesis against (ADR 0010, carrying forward ADR 0001 decision 1). "
    "This is reported as untestable, not resolved, and no Haitian-inclusive "
    "multilingual base is substituted to manufacture a testable stand-in."
)

# Readiness recommendation-rule thresholds (this ticket's "Proposed
# resolution" item 3: "thresholds are Open-Q module constants"). No source
# document ratifies specific numbers for the readiness recommendation rule
# itself (distinct from the red-team severity thresholds in
# docs/adr/0008-*.md Open Q2, which this rule does not reuse directly) — the
# values below are this ticket's own placeholder judgment, pending Co-PI
# ratification, exactly like coreset.PARTIAL_THRESHOLD_FRACTION's own
# "Revisit once project management sets a real number" precedent
# (src/coreset/__init__.py:64).
#
# READINESS_COVERAGE_MET_FRACTION / READINESS_COVERAGE_PARTIAL_FRACTION:
# the fraction of a scorecard's floor_verdicts that must be "met" (resp.
# at least "partial", i.e. not "unmet") for the coverage half of the
# go/partial/no_go rule below.
READINESS_COVERAGE_MET_FRACTION = 1.0  # Open Q (placeholder): all floor targets met
READINESS_COVERAGE_PARTIAL_FRACTION = 0.5  # Open Q (placeholder): at least half not-unmet

# READINESS_RELEASE_CLASSES_FOR_GO / _FOR_PARTIAL: which of the winning
# candidate's ModelReleaseClass values count toward a "go" vs "partial"
# recommendation. `internal_only`/`withdrawn` never contribute to either
# (a closed-tier or withdrawn winner cannot support any positive
# recommendation) — this is a placeholder pending ratification, same as
# the fractions above.
READINESS_RELEASE_CLASSES_FOR_GO: frozenset[ModelReleaseClass] = frozenset({"release_ready"})
READINESS_RELEASE_CLASSES_FOR_PARTIAL: frozenset[ModelReleaseClass] = frozenset(
    {"release_ready", "research_only"}
)

ReadinessRecommendation = str  # "go" | "partial" | "no_go" (kept a plain str: see below)


@dataclass(frozen=True, slots=True)
class LouMtArmResult:
    """One arm of the `lou` MT bake-off (ADR 0010 decision 2: Kreyòl-MT vs.
    the French-native LLM adapter vs. `facebook/mbart-large-50-many-to-many-mmt`
    as a mandatory no-creole control). Not defined in `src/bakeoff/` — the
    MT bake-off is a distinct three-arm harness from the generative
    bake-off `run_bakeoff`/`BakeoffRunResult` orchestrate (ADR 0010: "The
    `lou` bake-off becomes two separate harnesses"), and no sibling ticket
    in this wave builds it (Wave 3 plan on issue #23: nothing wires a
    `LouMtBakeoffResult`/`LouMtArmResult` producer this wave) — this
    module defines the minimal shape its own reader (`build_readiness_
    evidence`) needs, as an explicit input type a future MT-bake-off
    ticket constructs."""

    arm_id: str
    chrf: float
    bleu: float
    is_no_creole_control: bool


@dataclass(frozen=True, slots=True)
class LouMtBakeoffResult:
    """The `lou` MT bake-off's outcome (ADR 0010 decisions 2/3/5): every
    arm's chrF/BLEU (chrF primary per ADR 0010 decision 3), the winning
    `arm_id`, and whether the tie-break rule (ADR 0010 decision 5, `Open
    Q13`: ties go to the French-native arm on CARE grounds) was applied to
    reach that winner. `winner_arm_id` is `None`, never an arbitrary pick,
    when no arm can be recommended (mirrors `BakeoffRunResult.
    winner_candidate_id`'s own "None, never an arbitrary pick" contract,
    `src/bakeoff/__init__.py`)."""

    arms: tuple[LouMtArmResult, ...]
    winner_arm_id: str | None
    tie_break_applied: bool


@dataclass(frozen=True, slots=True)
class ReadinessEvidence:
    """The language-readiness evidence file (tech-spec v2 §7 bullet 4,
    filename `readiness_evidence_<language>.md`/`.json`): a go/partial/
    no_go recommendation generated from the coverage scorecard and the
    bake-off/eval results. `recommendation` is always one of "go" |
    "partial" | "no_go" (validated in `build_readiness_evidence`, not
    re-validated here — this dataclass trusts its own builder, matching
    `CoverageScorecard`'s own no-`__post_init__`-revalidation posture for
    its already-validated `TargetVerdict`/`CoverageStatus` fields).
    `hat_untestable_statement` always equals `HAT_UNTESTABLE_STATEMENT`
    verbatim for a `lou` file (enforced in `build_readiness_evidence`);
    `None` for `frc` (the `hat`-untestable finding is `lou`-specific — ADR
    0010 concerns `lou`'s ancestor-transfer hypothesis only, `frc` has no
    `hat`-transfer arm to report on)."""

    language: TargetLanguage
    run_id: str
    recommendation: ReadinessRecommendation
    recommendation_reason: str
    hat_untestable_statement: str | None
    lou_mt_result: LouMtBakeoffResult | None
    coverage_summary: FrozenMapping
    unmet_cells: tuple[str, ...]
    winner_candidate_id: str | None
    winner_release_class: ModelReleaseClass | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "run_id": self.run_id,
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
            "hat_untestable_statement": self.hat_untestable_statement,
            "lou_mt_result": (
                {
                    "arms": [
                        {
                            "arm_id": arm.arm_id,
                            "chrf": arm.chrf,
                            "bleu": arm.bleu,
                            "is_no_creole_control": arm.is_no_creole_control,
                        }
                        for arm in self.lou_mt_result.arms
                    ],
                    "winner_arm_id": self.lou_mt_result.winner_arm_id,
                    "tie_break_applied": self.lou_mt_result.tie_break_applied,
                }
                if self.lou_mt_result is not None
                else None
            ),
            "coverage_summary": dict(self.coverage_summary),
            "unmet_cells": list(self.unmet_cells),
            "winner_candidate_id": self.winner_candidate_id,
            "winner_release_class": self.winner_release_class,
        }

    def render_markdown(self) -> str:
        lines = [
            f"# Language-readiness evidence: {self.language}",
            "",
            f"- **run_id**: {self.run_id}",
            f"- **recommendation**: {self.recommendation}",
            f"- **reason**: {self.recommendation_reason}",
            f"- **winning candidate**: {_not_available(self.winner_candidate_id)}",
            f"- **winning candidate release class**: {_not_available(self.winner_release_class)}",
            "",
        ]
        if self.hat_untestable_statement is not None:
            lines.extend(["## `hat` capability arm", "", self.hat_untestable_statement, ""])
        if self.lou_mt_result is not None:
            lines.extend([
                "## `lou` MT bake-off result",
                "",
                "| arm_id | chrF | BLEU | no-creole control |",
                "|---|---|---|---|",
            ])
            for arm in self.lou_mt_result.arms:
                lines.append(
                    f"| {arm.arm_id} | {arm.chrf} | {arm.bleu} | {arm.is_no_creole_control} |"
                )
            lines.append("")
            lines.append(f"- winner: {_not_available(self.lou_mt_result.winner_arm_id)}")
            lines.append(f"- tie-break applied: {self.lou_mt_result.tie_break_applied}")
            lines.append("")
        lines.extend([
            "## Coverage summary",
            "",
        ])
        for key, value in sorted(self.coverage_summary.items()):
            lines.append(f"- {key}: {value}")
        lines.extend([
            "",
            "## Unmet diff-catalog cells",
            "",
        ])
        lines.extend(f"- {cell_id}" for cell_id in self.unmet_cells)
        return "\n".join(lines) + "\n"


def _readiness_recommendation(
    scorecard: CoverageScorecard, winner_release_class: ModelReleaseClass | None
) -> tuple[ReadinessRecommendation, str]:
    """The documented go/partial/no_go rule (this ticket's "Proposed
    resolution" item 3): computed from the scorecard's floor-target
    coverage status and the winning candidate's `release_class` — never
    from the candidate's raw score, so a high-scoring but ungoverned
    (e.g. `internal_only`) winner cannot force a "go".

    Rule: let `met_fraction` be the fraction of `scorecard.floor_verdicts`
    equal to "met" (0.0 for an empty scorecard). "go" requires both
    `met_fraction >= READINESS_COVERAGE_MET_FRACTION` and
    `winner_release_class in READINESS_RELEASE_CLASSES_FOR_GO`. Failing
    that, "partial" requires both `met_fraction >=
    READINESS_COVERAGE_PARTIAL_FRACTION` (i.e. at least half of floor
    targets are "met", the rest may be "partial" or "unmet") and
    `winner_release_class in READINESS_RELEASE_CLASSES_FOR_PARTIAL`.
    Otherwise "no_go". `no_go`/`partial` are first-class, honestly reported
    outcomes (tech-spec v2 §7, PRD §3) — this function never narrows scope
    to manufacture a "go"."""
    verdicts = list(scorecard.floor_verdicts.values())
    met_fraction = (
        sum(1 for v in verdicts if v == "met") / len(verdicts) if verdicts else 0.0
    )

    if (
        met_fraction >= READINESS_COVERAGE_MET_FRACTION
        and winner_release_class in READINESS_RELEASE_CLASSES_FOR_GO
    ):
        return (
            "go",
            f"floor coverage met_fraction={met_fraction:.2f} >= "
            f"{READINESS_COVERAGE_MET_FRACTION} and winner release_class="
            f"{winner_release_class!r} is release-ready.",
        )

    if (
        met_fraction >= READINESS_COVERAGE_PARTIAL_FRACTION
        and winner_release_class in READINESS_RELEASE_CLASSES_FOR_PARTIAL
    ):
        return (
            "partial",
            f"floor coverage met_fraction={met_fraction:.2f} >= "
            f"{READINESS_COVERAGE_PARTIAL_FRACTION} and winner release_class="
            f"{winner_release_class!r} supports a partial recommendation, "
            f"but full-go thresholds are not met.",
        )

    return (
        "no_go",
        f"floor coverage met_fraction={met_fraction:.2f} and winner "
        f"release_class={winner_release_class!r} do not clear the partial "
        "threshold.",
    )


def build_readiness_evidence(
    *,
    language: TargetLanguage,
    scorecard: CoverageScorecard,
    bakeoff_result: BakeoffRunResult,
    acceptance_reports: Mapping[Layer, AcceptanceReport],
    eval_report: Mapping[str, object],
    lou_mt_result: LouMtBakeoffResult | None = None,
) -> ReadinessEvidence:
    """Builds the language-readiness evidence file for `language` (tech-spec
    v2 §7 bullet 4).

    Raises `GovernanceArtifactError` if `language == "lou"` and
    `lou_mt_result is None` (this ticket's "Proposed resolution" item 3:
    "`build_readiness_evidence(language='lou', lou_mt_result=None)`
    raises" — a `lou` readiness file must always report the MT bake-off
    result; it cannot be silently omitted). Raises if `bakeoff_result.
    language != language` (a readiness file must not silently report a
    different language's bake-off result under this file's own language
    header) and if `scorecard.language != language`, for the same reason.
    Raises on `eval_report["run_id"]` disagreeing with `bakeoff_result`'s
    winner's `RunMetadata` is NOT checked here — this builder receives no
    `RunMetadata` of its own; the run-id/manifest-hash cross-check is
    `build_model_card`'s job (the model card is generated for the same
    `run_id` this readiness file's `bakeoff_result`/`eval_report` describe,
    so an inconsistency there is already caught before this file would be
    generated from the same inputs in the normal pipeline sequence).

    `hat_untestable_statement` is set to `HAT_UNTESTABLE_STATEMENT`
    verbatim for `language == "lou"`, `None` for `frc` (that finding is
    `lou`-specific).

    The recommendation is computed by `_readiness_recommendation` from
    `scorecard`'s floor-target coverage and the winning candidate's
    `release_class` (looked up from `bakeoff_result.results` by
    `bakeoff_result.winner_candidate_id`; `None` when there is no winner,
    e.g. every candidate disqualified — the "no model passes" contingency,
    ADR 0008 decision 5). "no_go"/"partial" are rendered as first-class
    outcomes on `ReadinessEvidence`, never narrowed into a smaller-scope
    "go" (tech-spec v2 §7, PRD §3)."""
    if language == "lou" and lou_mt_result is None:
        raise GovernanceArtifactError(
            "build_readiness_evidence(language='lou', ...) requires lou_mt_result "
            "(ADR 0010: the lou MT bake-off result is a mandatory honesty "
            "requirement of the lou readiness file, never silently omitted)"
        )
    if bakeoff_result.language != language:
        raise GovernanceArtifactError(
            f"bakeoff_result.language={bakeoff_result.language!r} does not match "
            f"language={language!r}"
        )
    if scorecard.language != language:
        raise GovernanceArtifactError(
            f"scorecard.language={scorecard.language!r} does not match language={language!r}"
        )

    winner_id = bakeoff_result.winner_candidate_id
    winner_release_class: ModelReleaseClass | None = None
    if winner_id is not None:
        for result in bakeoff_result.results:
            if result.candidate_id == winner_id:
                winner_release_class = result.release_class
                break

    recommendation, reason = _readiness_recommendation(scorecard, winner_release_class)

    eval_run_id = eval_report.get("run_id")
    run_id = str(eval_run_id) if eval_run_id is not None else ""

    coverage_summary = FrozenMapping(
        {
            "observed_items": scorecard.observed_counts.get("items", 0),
            "observed_speakers": scorecard.observed_counts.get("speakers", 0),
            "floor_verdicts_met": sum(1 for v in scorecard.floor_verdicts.values() if v == "met"),
            "floor_verdicts_total": len(scorecard.floor_verdicts),
        }
    )

    return ReadinessEvidence(
        language=language,
        run_id=run_id,
        recommendation=recommendation,
        recommendation_reason=reason,
        hat_untestable_statement=HAT_UNTESTABLE_STATEMENT if language == "lou" else None,
        lou_mt_result=lou_mt_result,
        coverage_summary=coverage_summary,
        unmet_cells=tuple(scorecard.unmet_cells),
        winner_candidate_id=winner_id,
        winner_release_class=winner_release_class,
    )


# --- write_artifacts (tech-spec v2 §7 filenames) ----------------------------


def write_artifacts(
    out_dir: Path,
    *,
    datasheet: Datasheet | None = None,
    model_card: ModelCard | None = None,
    readiness: ReadinessEvidence | None = None,
) -> dict[str, Path]:
    """Writes each supplied artifact as both `.md` (render_markdown()) and
    `.json` (json.dumps(to_dict())) under `out_dir`, using tech-spec v2
    §7's filename convention: `datasheet_<dataset_id>.{md,json}`,
    `model_card_<candidate_id>.{md,json}`,
    `readiness_evidence_<language>.{md,json}`. Creates `out_dir` if it does
    not exist. Returns a dict keyed `"<artifact>_md"`/`"<artifact>_json"`
    (e.g. `"datasheet_md"`) to every path actually written — an artifact
    left `None` contributes no keys, so a caller can tell exactly which
    files this call produced."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if datasheet is not None:
        md_path = out_dir / f"datasheet_{datasheet.dataset_id}.md"
        json_path = out_dir / f"datasheet_{datasheet.dataset_id}.json"
        md_path.write_text(datasheet.render_markdown(), encoding="utf-8")
        json_path.write_text(json.dumps(datasheet.to_dict(), indent=2) + "\n", encoding="utf-8")
        written["datasheet_md"] = md_path
        written["datasheet_json"] = json_path

    if model_card is not None:
        md_path = out_dir / f"model_card_{model_card.candidate_id}.md"
        json_path = out_dir / f"model_card_{model_card.candidate_id}.json"
        md_path.write_text(model_card.render_markdown(), encoding="utf-8")
        json_path.write_text(json.dumps(model_card.to_dict(), indent=2) + "\n", encoding="utf-8")
        written["model_card_md"] = md_path
        written["model_card_json"] = json_path

    if readiness is not None:
        md_path = out_dir / f"readiness_evidence_{readiness.language}.md"
        json_path = out_dir / f"readiness_evidence_{readiness.language}.json"
        md_path.write_text(readiness.render_markdown(), encoding="utf-8")
        json_path.write_text(json.dumps(readiness.to_dict(), indent=2) + "\n", encoding="utf-8")
        written["readiness_evidence_md"] = md_path
        written["readiness_evidence_json"] = json_path

    return written
