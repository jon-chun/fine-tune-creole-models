"""Coverage scorecard: the measurement layer for coreset selection (tech-spec §8).

Builds tech-spec §8's coverage scorecard — what fraction of each language's
floor/aspirational collection targets and diff-catalog cells are covered by
the current eligible item pool. This module does NOT implement the coreset
*selection* algorithm itself (k-center/DPP/MMR-style uncertainty×diversity
sampling over linguistic/lexical/categorical features) — that stays an
explicit seam for a future ticket, same "seam, not implementation" pattern
as src/bakeoff/'s fine_tune/score/run_red_team callables. This module's job
is producing the priority/coverage signal that selection algorithm (and
tech-spec §7's language-readiness evidence file) will consume.

diff_catalog_coverage is keyed by each cell's real `id` (e.g. ASP-001,
PRO-002) — the scheme actually used in configs/diff_catalog/*.yml — not the
illustrative slug-style keys in the tech-spec's example JSON. Each cell's
coverage_status is read verbatim from the diff-catalog YAML, not
re-inferred from item stratification: the diff-catalog file is the
linguist-facing source of truth for whether a construction is covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from data_contract import DataItem, LanguageTag, is_eligible

Priority = Literal["critical", "high", "medium", "low"]
CoverageStatus = Literal["met", "partial", "unmet"]

_PRIORITY_RANK: dict[Priority, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass(frozen=True, slots=True)
class DiffCatalogCell:
    """A typed read of one cell from configs/diff_catalog/*.yml's `cells:`
    list. Loading is in scope here; this module never writes back to the
    diff-catalog YAML files."""

    id: str
    axis: str
    feature: str
    priority: Priority
    coverage_status: CoverageStatus


@dataclass(frozen=True, slots=True)
class CoverageTargets:
    """tech-spec §8's targets.floor/targets.aspirational shape. Each side is
    a dict, matching the tech-spec's example JSON's free-form
    {"types": int, "speakers": int, ...} shape rather than a fixed set of
    named fields — the tech-spec doesn't close this set."""

    floor: dict[str, int]
    aspirational: dict[str, int]


@dataclass(frozen=True, slots=True)
class CoverageScorecard:
    """The typed equivalent of tech-spec §8's example JSON, extended with
    observed_counts (not in the tech-spec's illustrative example, but
    required for targets to be verifiable rather than a pass-through echo):
    the eligible item pool's actual types/speakers counts for `language`,
    directly comparable against targets.floor/targets.aspirational.
    "types" counts distinct item_ids; "speakers" counts distinct non-null
    `lect` values as a best-effort proxy — no dedicated speaker-identity
    field exists in the current data contract."""

    language: LanguageTag
    schema_version: str
    targets: CoverageTargets
    observed_counts: dict[str, int]
    diff_catalog_coverage: dict[str, CoverageStatus]
    unmet_cells: list[str]
    next_collection_priorities: list[str]
    annotation_hours_committed: float
    annotation_hours_budget_note: str


def load_diff_catalog_cells(path: Path) -> list[DiffCatalogCell]:
    """Parse one configs/diff_catalog/*.yml file's `cells:` list into typed
    DiffCatalogCell objects. Read-only — never modifies the source file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_cells = raw.get("cells", [])
    return [
        DiffCatalogCell(
            id=cell["id"],
            axis=cell["axis"],
            feature=cell["feature"],
            priority=cell["priority"],
            coverage_status=cell["coverage_status"],
        )
        for cell in raw_cells
    ]


def compute_coverage_scorecard(
    language: LanguageTag,
    items: list[DataItem],
    targets: CoverageTargets,
    diff_catalog_cells: list[DiffCatalogCell],
    *,
    annotation_hours_committed: float,
    annotation_hours_budget_note: str,
    schema_version: str = "1.0.0",
) -> CoverageScorecard:
    """Pure computation: no file or network I/O. `items` is filtered to
    is_eligible()-true items of `language` before counting toward coverage
    (tech-spec §2: ineligible items never consume annotation hours or enter
    a training/eval split — this scorecard honors that same rule).

    `diff_catalog_coverage` is read verbatim from each cell's own
    coverage_status. `unmet_cells` is exactly the unmet-status cell ids, in
    diff_catalog_cells' given order. `next_collection_priorities` orders
    unmet_cells by priority (critical > high > medium > low), ties broken by
    id.
    """
    eligible_items = [
        item for item in items if item.language_tag == language and is_eligible(item).eligible
    ]
    observed_counts = {
        "types": len({item.item_id for item in eligible_items}),
        "speakers": len({item.lect for item in eligible_items if item.lect is not None}),
    }

    diff_catalog_coverage: dict[str, CoverageStatus] = {
        cell.id: cell.coverage_status for cell in diff_catalog_cells
    }
    unmet_cells = [cell.id for cell in diff_catalog_cells if cell.coverage_status == "unmet"]

    unmet_lookup = {cell.id: cell for cell in diff_catalog_cells if cell.coverage_status == "unmet"}
    next_collection_priorities = sorted(
        unmet_cells,
        key=lambda cell_id: (_PRIORITY_RANK[unmet_lookup[cell_id].priority], cell_id),
    )

    return CoverageScorecard(
        language=language,
        schema_version=schema_version,
        targets=targets,
        observed_counts=observed_counts,
        diff_catalog_coverage=diff_catalog_coverage,
        unmet_cells=unmet_cells,
        next_collection_priorities=next_collection_priorities,
        annotation_hours_committed=annotation_hours_committed,
        annotation_hours_budget_note=annotation_hours_budget_note,
    )
