"""Coverage scorecard: the measurement layer for coreset selection (tech-spec
v2 §8).

Builds tech-spec v2 §8's coverage scorecard — what fraction of each
language's floor/aspirational collection targets and diff-catalog cells are
covered by the current eligible item pool. This module does NOT implement
the coreset *selection* algorithm itself (k-center/DPP/MMR-style
uncertainty×diversity sampling over linguistic/lexical/categorical
features) — that stays an explicit seam for a future ticket (backlog 0012),
same "seam, not implementation" pattern as src/bakeoff/'s
fine_tune/score/run_red_team callables. This module's job is producing the
priority/coverage signal that selection algorithm (and tech-spec v2 §7's
language-readiness evidence file) will consume.

diff_catalog_coverage is keyed by each cell's real `id` (e.g. ASP-001,
PRO-002) — the scheme actually used in configs/diff_catalog/*.yml — not the
illustrative slug-style keys in the tech-spec's example JSON. Each cell's
coverage_status is read verbatim from the diff-catalog YAML, not
re-inferred from item stratification: the diff-catalog file is the
linguist-facing source of truth for whether a construction is covered
(ADR 0003, unchanged by this v2 migration).

Contract v2 migration (MIG-01f, issue #30): `DiffCatalogCell` gains the v2
cell fields tech-spec v2 §4 (D-6) requires (`gate_class`, `ortho_visible`,
`modality`, `lect_scope`, `probe_task`, `base_failure_rate`, `min_probes`,
and the phon-file-only `environment`/`respelling_attested`);
`compute_coverage_scorecard`'s `observed_counts["speakers"]` is now a real
`speaker_id`-based count (MIG-01a added `DataItem.speaker_id`), not the
`lect`-based proxy this module's docstring used to flag as a stand-in; the
scorecard gains `gate_class`/`base_failure_rate` maps and optional
stratification by `lect`/`genre`/`speaker_generation`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from data_contract import DataItem, LanguageTag, is_eligible, validate_literal

Priority = Literal["critical", "high", "medium", "low"]
CoverageStatus = Literal["met", "partial", "unmet"]

_PRIORITIES = frozenset(get_args(Priority))
_COVERAGE_STATUSES = frozenset(get_args(CoverageStatus))

_PRIORITY_RANK: dict[Priority, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# TargetVerdict threshold (issue #18): a target counts as "partial" once
# observed reaches at least this fraction of the target value, short of
# "met" (observed >= target). No source document names a number for this —
# the tech-spec's §8 JSON example shows a "partial" status on
# diff_catalog_coverage cells (linguist-authored, not derived) but never
# defines a partial threshold for the floor/aspirational targets comparison
# story 6 asked for. 0.5 is the named, documented choice for MVP 0.1: below
# half of a target is "unmet" (not meaningfully progressing), at/above half
# but short of the full target is "partial" (visibly progressing), and
# reaching or exceeding the target is "met". Revisit once project management
# sets a real number.
PARTIAL_THRESHOLD_FRACTION = 0.5

TargetVerdict = Literal["met", "partial", "unmet"]
_TARGET_VERDICTS = frozenset(get_args(TargetVerdict))

# --- v2 diff-catalog cell fields (tech-spec v2 §4, D-6) ---------------------

# gate_class: §6.3's release-gate severity class, independent of `priority`
# (tech-spec v2 §4: "priority (drives annotation and synthesis; unchanged) ·
# gate_class 0-3 (§6.3; independent of priority)"; §6.3: "The two are set
# independently by the linguist."). `None` is a valid, distinct state (the
# phon-file P2 worked example ships `gate_class: null` with the comment "no
# stop rule in 0.1.5") — never conflated with any of 0-3.
_GATE_CLASSES = frozenset({0, 1, 2, 3})

# modality: tech-spec v2 §4 "modality text | speech".
Modality = Literal["text", "speech"]
_MODALITIES = frozenset(get_args(Modality))

# probe_task: tech-spec v2 §4's closed set verbatim ("probe_task normalize |
# contrast | judge | translate_eng | translate_fra | phone_error | g2p |
# alignment"). The tech-spec's own phon-file worked example (§4 excerpt,
# `frc_vs_fra_phon.yml`) spells this key `probe_type` instead of
# `probe_task` on that one line — a source-document inconsistency, not a
# second, different field: the general D-6 field list, the frc_vs_fra.yml
# VERB-001 example, and the lou_vs_hat.yml ANTI-HAT-001 instruction all use
# `probe_task`. Judgment call: `probe_task` is the one canonical
# `DiffCatalogCell` field; `load_diff_catalog_cells` accepts a YAML cell
# spelled either `probe_task` or `probe_type` (the latter normalized to the
# former at load time) so the tech-spec's own example YAML still loads
# without edits.
ProbeTask = Literal[
    "normalize", "contrast", "judge", "translate_eng", "translate_fra",
    "phone_error", "g2p", "alignment",
]
_PROBE_TASKS = frozenset(get_args(ProbeTask))

# respelling_attested: phon-file-only field. The tech-spec v2 §4 prose lists
# the type as "true | false | unknown" — a three-state value a plain `bool`
# cannot represent — but its own worked YAML example spells the value as a
# bare YAML bool (`respelling_attested: true`). Judgment call (per the
# ticket's recommendation): model this as a closed string-Literal 3-state
# enum on `DiffCatalogCell`, and normalize a YAML bool at load time to its
# string equivalent (`True -> "true"`, `False -> "false"`) so the tech-spec's
# own example YAML loads unchanged.
RespellingAttested = Literal["true", "false", "unknown"]
_RESPELLING_ATTESTED_VALUES = frozenset(get_args(RespellingAttested))


class CoresetConfigError(ValueError):
    """Raised when a configs/diff_catalog/*.yml cell is malformed: a missing
    required key, or a priority/coverage_status/gate_class/etc. value
    outside its closed set (issue #15)."""


def _normalize_lect_scope(value: object) -> tuple[str, ...] | None:
    """Normalize a diff-catalog cell's `lect_scope` to a tuple regardless of
    whether the YAML source spells it as a bare string (`lect_scope: all`)
    or a list (`lect_scope: [acadiana_west_prairie, acadiana_central]`) —
    tech-spec v2 §4's own two worked examples use both spellings. `None`
    passes through unchanged. A bare string becomes a one-element tuple
    (never split on whitespace/commas: `"all"` is itself one scope value,
    not a delimited list), so every consumer can iterate `lect_scope`
    uniformly without a per-call-site type check (user story 12)."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise CoresetConfigError(f"lect_scope={value!r} is not a string or list")


def _normalize_respelling_attested(value: Any) -> Any:
    """Normalize `respelling_attested` to the closed `RespellingAttested`
    string set. Accepts the tech-spec's own YAML spelling (a bare bool,
    `true`/`false`) and maps it to the string equivalent; accepts the
    three-state strings directly (validated by `DiffCatalogCell.__post_init__`,
    not here — this function only normalizes shape); `None` passes through
    unchanged. Typed `Any` in/out, matching every other raw-YAML-value site
    in `load_diff_catalog_cells` (the parsed YAML cell itself is `Any` —
    `__post_init__` is what actually enforces closed-set membership at
    runtime, not the type system, same split as `data_contract.read_manifest`)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    raise CoresetConfigError(f"respelling_attested={value!r} is not a bool or string")


@dataclass(frozen=True, slots=True)
class DiffCatalogCell:
    """A typed read of one cell from configs/diff_catalog/*.yml's `cells:`
    list. Loading is in scope here; this module never writes back to the
    diff-catalog YAML files.

    Keeps every field actually present across the three v1 catalogs
    (issue #18: frc_form/fra_contrast/failure_mode/source/note/etc.) plus
    the v2 cell fields tech-spec v2 §4 (D-6) requires: `gate_class`,
    `ortho_visible`, `modality`, `lect_scope`, `probe_task`,
    `base_failure_rate`, `min_probes`, and the phon-file-only `environment`/
    `respelling_attested`. Every field beyond the five originally required
    (`id`/`axis`/`feature`/`priority`/`coverage_status`) is optional
    (`| None`, default None) since no single cell across any catalog file
    carries every one of them (this dataclass's own long-standing
    convention, unchanged by this migration)."""

    id: str
    axis: str
    feature: str
    priority: Priority
    coverage_status: CoverageStatus
    failure_mode: str | None = None
    source: str | None = None
    note: str | None = None
    # frc_vs_fra.yml / lou_vs_fra.yml surface forms:
    frc_form: str | None = None
    lou_form: str | None = None
    fra_contrast: str | None = None
    # lou_vs_hat.yml anti-conflation-axis surface forms:
    shared_surface_form: str | None = None
    why_convergent_not_derived: str | None = None
    # v2 cell fields (tech-spec v2 §4, D-6):
    gate_class: int | None = None
    ortho_visible: bool | None = None
    modality: Modality | None = None
    lect_scope: tuple[str, ...] | None = None
    probe_task: tuple[ProbeTask, ...] | None = None
    base_failure_rate: float | None = None
    # min_probes: no repo-wide default is set here (issue #30 "Proposed
    # resolution": "implement the default at the scorecard/probe-authoring
    # layer, not silently defaulted inside DiffCatalogCell itself," matching
    # MIG-01a's `data_class` no-invented-default posture) — `None` here
    # means "use the tech-spec v2 §4 default of 40 (80 with the extension)"
    # at whatever probe-authoring/scorecard call site consumes it; this
    # dataclass never substitutes that default silently.
    min_probes: int | None = None
    # phon-file-only fields (tech-spec v2 §4 "phon file only"):
    environment: dict[str, str] | None = None
    respelling_attested: RespellingAttested | None = None

    def __post_init__(self) -> None:
        # lect_scope normalization (user story 12) is applied on every
        # construction path, not just load_diff_catalog_cells — mirrors
        # data_contract.DataItem's own __post_init__ alias-normalization
        # pattern (object.__setattr__ is the standard escape hatch a frozen
        # dataclass's own __post_init__ uses to mutate itself). A caller
        # constructing DiffCatalogCell directly with a bare-string
        # lect_scope gets the same normalized tuple a YAML-loaded cell does.
        if self.lect_scope is not None and isinstance(self.lect_scope, str):
            object.__setattr__(self, "lect_scope", (self.lect_scope,))
        elif self.lect_scope is not None and not isinstance(self.lect_scope, tuple):
            object.__setattr__(self, "lect_scope", tuple(self.lect_scope))

        # respelling_attested normalization: a caller passing the tech-spec's
        # bare-YAML-bool spelling directly gets the same string mapping
        # load_diff_catalog_cells applies.
        if isinstance(self.respelling_attested, bool):
            object.__setattr__(
                self, "respelling_attested", "true" if self.respelling_attested else "false"
            )

        # Same enum-membership discipline as DataItem's own __post_init__
        # (issue #15): a bad priority/coverage_status/gate_class/etc. value
        # must be rejected at construction, not silently accepted and only
        # noticed later as a bare KeyError out of _PRIORITY_RANK (tech-spec
        # review R6).
        validate_literal(self.priority, tuple(_PRIORITIES), "priority")
        validate_literal(self.coverage_status, tuple(_COVERAGE_STATUSES), "coverage_status")

        if self.gate_class is not None and self.gate_class not in _GATE_CLASSES:
            raise CoresetConfigError(
                f"gate_class={self.gate_class!r} is not one of {sorted(_GATE_CLASSES)} or None"
            )
        if self.modality is not None:
            validate_literal(self.modality, tuple(_MODALITIES), "modality")
        if self.probe_task is not None:
            for task in self.probe_task:
                validate_literal(task, tuple(_PROBE_TASKS), "probe_task")
        if self.respelling_attested is not None:
            validate_literal(
                self.respelling_attested, tuple(_RESPELLING_ATTESTED_VALUES), "respelling_attested"
            )


@dataclass(frozen=True, slots=True)
class CoverageTargets:
    """tech-spec v2 §8's targets.floor/targets.aspirational shape. Each side
    is a dict, matching the tech-spec's example JSON's free-form
    {"items": int, "speakers": int, ...} shape rather than a fixed set of
    named fields — the tech-spec doesn't close this set."""

    floor: dict[str, int]
    aspirational: dict[str, int]


@dataclass(frozen=True, slots=True)
class CoverageScorecard:
    """The typed equivalent of tech-spec v2 §8's example JSON, extended with
    observed_counts (not in the tech-spec's illustrative example, but
    required for targets to be verifiable rather than a pass-through echo):
    the eligible item pool's actual items/speakers counts for `language`,
    directly comparable against targets.floor/targets.aspirational.
    "items" counts distinct item_ids; "speakers" counts distinct non-null
    `speaker_id` values (MIG-01f: a real speaker-identity count, replacing
    the `lect`-based proxy this docstring used to flag as a stand-in — see
    CONTEXT.md/tech-spec v2 §8 patch note)."""

    language: LanguageTag
    schema_version: str
    targets: CoverageTargets
    observed_counts: dict[str, int]
    # target_verdicts (issue #18): per-target-key met/partial/unmet verdict,
    # one dict per side of CoverageTargets (floor/aspirational), keyed the
    # same as observed_counts/targets.floor/targets.aspirational themselves.
    floor_verdicts: dict[str, TargetVerdict]
    aspirational_verdicts: dict[str, TargetVerdict]
    diff_catalog_coverage: dict[str, CoverageStatus]
    # gate_class / base_failure_rate maps (MIG-01f, tech-spec v2 §8 example
    # JSON: "gate_class": {"VERB-001": 1, ...}, "base_failure_rate":
    # {"VERB-001": 0.82, ...}): populated from each DiffCatalogCell's own
    # gate_class/base_failure_rate values. A cell whose gate_class or
    # base_failure_rate is None is omitted from the respective map entirely
    # (never included with a None/null placeholder value), keeping both
    # maps' value types strictly int/float per the ticket's "Proposed
    # resolution".
    gate_class: dict[str, int]
    base_failure_rate: dict[str, float]
    unmet_cells: list[str]
    next_collection_priorities: list[str]
    annotation_hours_committed: float
    annotation_hours_budget_note: str
    # stratified_counts (MIG-01f, tech-spec v2 §8 "stratifies on
    # language_tag/lect/genre/speaker_generation"): an additional, optional
    # breakdown of observed_counts by one or more of
    # {lect, genre, speaker_generation}, keyed "<stratify-field>:<value>" ->
    # {"items": n, "speakers": m}. Empty when compute_coverage_scorecard's
    # stratify_by parameter is not given — preserves backward-shape
    # compatibility for every existing observed_counts consumer while adding
    # the new per-axis breakdown tech-spec v2 requires (issue #30 "Proposed
    # resolution").
    stratified_counts: dict[str, dict[str, int]]


def _target_verdict(observed: int, target: int) -> TargetVerdict:
    """met: observed >= target. unmet: observed below
    PARTIAL_THRESHOLD_FRACTION of target. partial: everything between.
    A target of 0 is trivially met by any non-negative observed count
    (there is nothing left to collect)."""
    if target <= 0:
        return "met"
    if observed >= target:
        return "met"
    if observed >= target * PARTIAL_THRESHOLD_FRACTION:
        return "partial"
    return "unmet"


def _target_verdicts(observed_counts: dict[str, int], target_side: dict[str, int]) -> dict[str, TargetVerdict]:
    return {
        key: _target_verdict(observed_counts.get(key, 0), target_value)
        for key, target_value in target_side.items()
    }


def load_diff_catalog_cells(path: Path) -> list[DiffCatalogCell]:
    """Parse one configs/diff_catalog/*.yml file's `cells:` list into typed
    DiffCatalogCell objects. Read-only — never modifies the source file.

    Raises CoresetConfigError — never a raw KeyError — if a cell is missing
    a required key or has a priority/coverage_status/gate_class/modality/
    probe_task/respelling_attested value outside its closed set (issue #15).
    Every field beyond the five required ones is read with `.get` and
    defaults to None when a given catalog file doesn't carry it (issue #18:
    the catalogs don't share one optional-field shape — see
    DiffCatalogCell's docstring).

    `lect_scope` is normalized to a tuple regardless of whether the source
    YAML spells it as a bare string or a list (`_normalize_lect_scope`).
    `respelling_attested` accepts either the tech-spec's bare-YAML-bool
    spelling or the three-state string directly (`_normalize_respelling_attested`).
    `probe_task` accepts a cell keyed either `probe_task` or `probe_type` —
    the tech-spec v2 §4 phon-file worked example spells this key
    `probe_type` on one line, a source-document inconsistency this loader
    tolerates rather than propagates (see the `ProbeTask` module comment)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_cells = raw.get("cells", [])
    cells: list[DiffCatalogCell] = []
    for cell in raw_cells:
        probe_task_raw = cell.get("probe_task", cell.get("probe_type"))
        try:
            cells.append(
                DiffCatalogCell(
                    id=cell["id"],
                    axis=cell["axis"],
                    feature=cell["feature"],
                    priority=cell["priority"],
                    coverage_status=cell["coverage_status"],
                    failure_mode=cell.get("failure_mode"),
                    source=cell.get("source"),
                    note=cell.get("note"),
                    frc_form=cell.get("frc_form"),
                    lou_form=cell.get("lou_form"),
                    fra_contrast=cell.get("fra_contrast"),
                    shared_surface_form=cell.get("shared_surface_form"),
                    why_convergent_not_derived=cell.get("why_convergent_not_derived"),
                    gate_class=cell.get("gate_class"),
                    ortho_visible=cell.get("ortho_visible"),
                    modality=cell.get("modality"),
                    lect_scope=_normalize_lect_scope(cell.get("lect_scope")),
                    probe_task=tuple(probe_task_raw) if probe_task_raw is not None else None,
                    base_failure_rate=cell.get("base_failure_rate"),
                    min_probes=cell.get("min_probes"),
                    environment=cell.get("environment"),
                    respelling_attested=_normalize_respelling_attested(
                        cell.get("respelling_attested")
                    ),
                )
            )
        except KeyError as exc:
            raise CoresetConfigError(f"{path}: cell missing required field {exc}") from exc
        except ValueError as exc:
            raise CoresetConfigError(f"{path}: {exc}") from exc
    return cells


# stratify_by: the closed set of DataItem fields compute_coverage_scorecard
# may additionally break observed_counts down by (tech-spec v2 §8:
# "stratifies on language_tag/lect/genre/speaker_generation" — language_tag
# is already the top-level scorecard axis, so only the remaining three are
# offered here).
StratifyField = Literal["lect", "genre", "speaker_generation"]
_STRATIFY_FIELDS = frozenset(get_args(StratifyField))


def compute_coverage_scorecard(
    language: LanguageTag,
    items: list[DataItem],
    targets: CoverageTargets,
    diff_catalog_cells: list[DiffCatalogCell],
    *,
    annotation_hours_committed: float,
    annotation_hours_budget_note: str,
    schema_version: str = "2.0.0",
    stratify_by: tuple[StratifyField, ...] | None = None,
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

    `floor_verdicts`/`aspirational_verdicts` (issue #18) compare
    `observed_counts` against each side of `targets` key-by-key, via
    `_target_verdict`: `met` (observed >= target), `unmet` (observed below
    PARTIAL_THRESHOLD_FRACTION of target), `partial` (between). A target key
    absent from `observed_counts` (e.g. a future non-items/speakers target
    key this pool doesn't track) is treated as an observed count of 0, never
    a KeyError.

    `gate_class`/`base_failure_rate` (MIG-01f) are populated from each
    `diff_catalog_cells` entry's own field of the same name; a cell with
    `gate_class`/`base_failure_rate` of `None` is omitted from the
    respective map (never included with a `None`/`null` value).

    `stratify_by` (MIG-01f; tech-spec v2 §8 "stratifies on ... lect/genre/
    speaker_generation"): when given, `stratified_counts` additionally
    breaks the eligible item pool down by each named `DataItem` field,
    keyed `"<field>:<value>"` -> `{"items": n, "speakers": m}` — one entry
    per distinct value of that field seen among the eligible items (items
    with a `None` value for that field are grouped under the literal key
    `"<field>:None"` rather than silently dropped). Backward-compatible:
    `observed_counts`'s own shape is unchanged whether or not `stratify_by`
    is given, so every existing consumer of `observed_counts["items"]`/
    `observed_counts["speakers"]` keeps working unmodified.
    """
    eligible_items = [
        item for item in items if item.language_tag == language and is_eligible(item).eligible
    ]
    observed_counts = {
        "items": len({item.item_id for item in eligible_items}),
        "speakers": len({item.speaker_id for item in eligible_items if item.speaker_id is not None}),
    }

    floor_verdicts = _target_verdicts(observed_counts, targets.floor)
    aspirational_verdicts = _target_verdicts(observed_counts, targets.aspirational)

    diff_catalog_coverage: dict[str, CoverageStatus] = {
        cell.id: cell.coverage_status for cell in diff_catalog_cells
    }
    unmet_cells = [cell.id for cell in diff_catalog_cells if cell.coverage_status == "unmet"]

    unmet_lookup = {cell.id: cell for cell in diff_catalog_cells if cell.coverage_status == "unmet"}
    next_collection_priorities = sorted(
        unmet_cells,
        key=lambda cell_id: (_PRIORITY_RANK[unmet_lookup[cell_id].priority], cell_id),
    )

    gate_class_map: dict[str, int] = {
        cell.id: cell.gate_class for cell in diff_catalog_cells if cell.gate_class is not None
    }
    base_failure_rate_map: dict[str, float] = {
        cell.id: cell.base_failure_rate
        for cell in diff_catalog_cells
        if cell.base_failure_rate is not None
    }

    stratified_counts: dict[str, dict[str, int]] = {}
    for field_name in stratify_by or ():
        if field_name not in _STRATIFY_FIELDS:
            raise CoresetConfigError(
                f"stratify_by field {field_name!r} is not one of {sorted(_STRATIFY_FIELDS)}"
            )
        buckets: dict[object, list[DataItem]] = {}
        for item in eligible_items:
            value = getattr(item, field_name)
            buckets.setdefault(value, []).append(item)
        for value, bucket_items in buckets.items():
            stratified_counts[f"{field_name}:{value}"] = {
                "items": len({item.item_id for item in bucket_items}),
                "speakers": len(
                    {item.speaker_id for item in bucket_items if item.speaker_id is not None}
                ),
            }

    return CoverageScorecard(
        language=language,
        schema_version=schema_version,
        targets=targets,
        observed_counts=observed_counts,
        floor_verdicts=floor_verdicts,
        aspirational_verdicts=aspirational_verdicts,
        diff_catalog_coverage=diff_catalog_coverage,
        gate_class=gate_class_map,
        base_failure_rate=base_failure_rate_map,
        unmet_cells=unmet_cells,
        next_collection_priorities=next_collection_priorities,
        annotation_hours_committed=annotation_hours_committed,
        annotation_hours_budget_note=annotation_hours_budget_note,
        stratified_counts=stratified_counts,
    )


def scorecard_to_dict(scorecard: CoverageScorecard) -> dict[str, Any]:
    """The JSON-serializable dict form of a CoverageScorecard (tech-spec §8:
    "JSON, machine-readable"; the augment CLI's `--coverage-scorecard <json>`
    input). A plain `dataclasses.asdict` already works here (every field is
    already JSON-primitive: str/float/dict[str,str|int]/list[str] — no
    datetime or other non-serializable type, unlike src.tracking.RunMetadata
    per D4-2), but this wrapper is the one blessed way to get the dict form,
    so `scorecard_from_json` has a single matching counterpart to invert."""
    return asdict(scorecard)


def scorecard_to_json(scorecard: CoverageScorecard) -> str:
    """`json.dumps(scorecard_to_dict(scorecard))`, the literal JSON text
    tech-spec §8 calls for."""
    return json.dumps(scorecard_to_dict(scorecard))


def scorecard_from_dict(data: dict[str, Any]) -> CoverageScorecard:
    """Inverse of scorecard_to_dict: rebuilds a CoverageScorecard (and its
    nested CoverageTargets) from a plain dict of the same shape."""
    targets_data = data["targets"]
    return CoverageScorecard(
        language=data["language"],
        schema_version=data["schema_version"],
        targets=CoverageTargets(floor=targets_data["floor"], aspirational=targets_data["aspirational"]),
        observed_counts=data["observed_counts"],
        floor_verdicts=data["floor_verdicts"],
        aspirational_verdicts=data["aspirational_verdicts"],
        diff_catalog_coverage=data["diff_catalog_coverage"],
        gate_class=data["gate_class"],
        base_failure_rate=data["base_failure_rate"],
        unmet_cells=data["unmet_cells"],
        next_collection_priorities=data["next_collection_priorities"],
        annotation_hours_committed=data["annotation_hours_committed"],
        annotation_hours_budget_note=data["annotation_hours_budget_note"],
        stratified_counts=data["stratified_counts"],
    )


def scorecard_from_json(text: str) -> CoverageScorecard:
    """Inverse of scorecard_to_json: parses JSON text back into a typed
    CoverageScorecard."""
    return scorecard_from_dict(json.loads(text))


__all__ = [
    "Priority",
    "CoverageStatus",
    "PARTIAL_THRESHOLD_FRACTION",
    "TargetVerdict",
    "Modality",
    "ProbeTask",
    "RespellingAttested",
    "StratifyField",
    "CoresetConfigError",
    "DiffCatalogCell",
    "CoverageTargets",
    "CoverageScorecard",
    "load_diff_catalog_cells",
    "compute_coverage_scorecard",
    "scorecard_to_dict",
    "scorecard_to_json",
    "scorecard_from_dict",
    "scorecard_from_json",
]
