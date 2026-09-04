"""Base-model bake-off: config loader + orchestration skeleton (contract v2).

tech-spec v2 §3.2: base-model selection is never assumed, always an empirical
bake-off. This module builds two real pieces (the config loader and the
result-aggregation orchestration in run_bakeoff) around four stub-shaped
seams — `fine_tune`, `score`, `score_untuned_base`, `run_red_team` — that a
later ticket fills in with real implementations:

- `fine_tune`: the LoRA/QLoRA/DoRA training driver (tech-spec v2 §5). Not
  built here. Injected as a callable so run_bakeoff never imports a concrete
  trainer. Generic over the caller's hyperparameters type `H` — this module
  never imports `train.Hyperparameters` (no cross-sibling import; issue #14,
  restated issue #27) so it cannot name that type concretely. A caller wires
  its own `Hyperparameters` in as `H` at the call site.
- `score`: the Layer-B gold-set scoring harness (tech-spec v2 §6). Not built
  here — no gold set exists yet either.
- `score_untuned_base`: scores each candidate's own untuned base on the same
  locked probes, before that candidate's fine_tune/score/run_red_team
  sequence (tech-spec v2 §3.2 harness responsibility 3). Not built here.
- `run_red_team`: the conflation red-team suite (tech-spec v2 §6.3). Not
  built here — backlog 0011.

Calling run_bakeoff() with placeholder/test-double callables exercises the
orchestration logic only — candidate iteration, disqualification precedence,
seed aggregation, winner selection — never a real bake-off result. This
module ships no training, scoring, or red-team logic of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import yaml

from data_contract import ModelReleaseClass, TargetLanguage

_SUPPORTED_LANGUAGES = frozenset({"frc", "lou"})

# H: the caller's hyperparameters type (e.g. train.Hyperparameters), threaded
# opaquely through run_bakeoff/fine_tune without this module importing it
# (no cross-sibling import — issue #14 ground rules, restated issue #27). H
# is not itself seeded/keyed on S below — every candidate/seed pair receives
# the identical H value (tech-spec v2 §3.2 harness responsibility 1).
H = TypeVar("H")


class BakeoffConfigError(ValueError):
    """Raised when bakeoff_candidates.yml is malformed or fails validation,
    including a release-eligible arm whose license is outside
    `defaults.release_licenses` (issue #27, promotes backlog 0015)."""


class BakeoffError(ValueError):
    """Raised when run_bakeoff() is called with a language absent from the
    supplied config, when candidates disagree on the scoring metric, or when
    `seeds` is empty."""


# --- Candidate / config shapes (tech-spec v2 §3.2 config surface) -----------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One bake-off arm. Contract v2: the `base: null` control-candidate
    concept (issue #14 R5) is removed entirely (issue #27) — control is now
    a `control: "untuned_base"` string on the language's config, naming each
    arm's own base scored first, not a fourth candidate row. Every
    `Candidate` therefore carries a real `hf_repo`/`license`.

    `size_b`/`size_m` are mutually exclusive size fields (billions of
    parameters for a generative arm, millions for the Kreyòl-MT MT arm);
    `instruct` defaults to `defaults.instruct` from the config's `defaults`
    block; `optional` marks an arm that is not required to run (e.g. the
    Claire-7B-Apache base-only F arm, Open Q9); `alt` records a documented
    alternative model id, never a second candidate to score (tech-spec v2
    §3.2 config surface). `base`/`lang_token` are MT-arm-only fields
    (Kreyòl-MT's mBART-50 backbone and language token); both `None` for a
    non-MT arm.
    """

    id: str
    hf_repo: str
    license: str
    size_b: int | None
    size_m: int | None
    instruct: bool
    optional: bool
    alt: str | None
    base: str | None
    lang_token: str | None
    note: str | None


@dataclass(frozen=True, slots=True)
class BakeoffDefaults:
    """The config's `defaults:` block (tech-spec v2 §3.2 config surface)."""

    instruct: bool
    seeds: int
    forgetting_axis: str
    release_licenses: tuple[str, ...]
    nf4_min_size_b: int | None


@dataclass(frozen=True, slots=True)
class FrcLanguageConfig:
    """The `frc:` section: a flat `arms:` list plus its `control` marker
    (tech-spec v2 §3.2 config surface)."""

    control: str
    arms: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class LouLanguageConfig:
    """The `lou:` section: two lanes (ADR 0010, R3) — `generative_arms` (2
    entries) and `mt_arms` (3 entries) — plus the free-text `prerequisite`
    (Open Q15) and `decision_rule` (Q13, closed 2026-09-03, ADR 0015) fields.
    Replaces ADR 0001's flat four-arm `candidates` list entirely."""

    control: str
    generative_arms: tuple[Candidate, ...]
    mt_arms: tuple[Candidate, ...]
    prerequisite: str
    decision_rule: str


@dataclass(frozen=True, slots=True)
class BakeoffConfig:
    """The full parsed `bakeoff_candidates.yml` (schema_version "2.0.0")."""

    schema_version: str
    defaults: BakeoffDefaults
    frc: FrcLanguageConfig
    lou: LouLanguageConfig

    def arms_for(self, language: TargetLanguage) -> tuple[Candidate, ...]:
        """The flat candidate list `run_bakeoff` iterates for `language`.
        `lou`'s two lanes are concatenated (generative_arms then mt_arms) —
        run_bakeoff treats every arm identically regardless of lane; lane
        membership is a config/reporting distinction (ADR 0010), not an
        orchestration one."""
        if language == "frc":
            return self.frc.arms
        return self.lou.generative_arms + self.lou.mt_arms


@dataclass(frozen=True, slots=True)
class LoadedBakeoffConfig:
    """A loaded `BakeoffConfig` plus any soft-gate warnings — currently only
    a `[VERIFY]`-tagged license (tech-spec v2 §3.2's re-fetch-at-bake-off-time
    note; issue #27 "License allowlist" proposed resolution). Mirrors the
    existing `LoadedTrainingConfig` soft-gate/hard-gate split in
    `src/train/__init__.py`'s `load_hyperparameters` (rank/epochs warn,
    quantization hard-rejects) — verified by reading that module's
    `LoadedTrainingConfig`/`load_hyperparameters` in full."""

    config: BakeoffConfig
    warnings: list[str]


def _parse_defaults(raw: dict[str, object], path: Path) -> BakeoffDefaults:
    if not isinstance(raw, dict):
        raise BakeoffConfigError(f"{path}: 'defaults' must be a mapping")

    release_licenses_raw = raw.get("release_licenses")
    if not isinstance(release_licenses_raw, list) or not release_licenses_raw:
        raise BakeoffConfigError(
            f"{path}: defaults.release_licenses must be a non-empty list "
            "(issue #27, promotes backlog 0015)"
        )
    if not all(isinstance(item, str) for item in release_licenses_raw):
        raise BakeoffConfigError(f"{path}: defaults.release_licenses entries must all be strings")

    seeds_raw = raw.get("seeds")
    if not isinstance(seeds_raw, int) or isinstance(seeds_raw, bool) or seeds_raw <= 0:
        raise BakeoffConfigError(f"{path}: defaults.seeds must be a positive int")

    instruct_raw = raw.get("instruct")
    # tech-spec v2 §3.2 writes `instruct: required` (a policy string), not a
    # bool — "required" means every arm defaults to instruct=True unless it
    # explicitly opts out (e.g. the optional Claire-7B-Apache base arm, Open
    # Q9). A literal bool is also accepted for a caller/test fixture that
    # prefers to spell it directly.
    if instruct_raw == "required" or instruct_raw is True:
        instruct_default = True
    elif instruct_raw is False:
        instruct_default = False
    else:
        raise BakeoffConfigError(
            f"{path}: defaults.instruct must be 'required' or a bool, got {instruct_raw!r}"
        )

    forgetting_axis_raw = raw.get("forgetting_axis")
    if not isinstance(forgetting_axis_raw, str) or not forgetting_axis_raw:
        raise BakeoffConfigError(f"{path}: defaults.forgetting_axis must be a non-empty string")

    nf4_min_size_b_raw = raw.get("nf4_min_size_b")
    nf4_min_size_b: int | None
    if nf4_min_size_b_raw is None:
        nf4_min_size_b = None
    elif isinstance(nf4_min_size_b_raw, int) and not isinstance(nf4_min_size_b_raw, bool):
        nf4_min_size_b = nf4_min_size_b_raw
    else:
        raise BakeoffConfigError(f"{path}: defaults.nf4_min_size_b must be an int or null")

    return BakeoffDefaults(
        instruct=instruct_default,
        seeds=seeds_raw,
        forgetting_axis=forgetting_axis_raw,
        release_licenses=tuple(release_licenses_raw),
        nf4_min_size_b=nf4_min_size_b,
    )


def _parse_optional_int(raw: dict[str, object], key: str, path: Path, arm_id: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise BakeoffConfigError(f"{path}: {arm_id}.{key} must be an int or null, got {value!r}")


def _parse_arm(
    raw: dict[str, object],
    *,
    language: str,
    path: Path,
    defaults: BakeoffDefaults,
    allowlist: frozenset[str],
    warnings: list[str],
) -> Candidate:
    """Parse one `arms`/`generative_arms`/`mt_arms` entry (issue #27
    "Proposed resolution"). Unlike the superseded `_parse_candidate`, there
    is no `base: null` control special-case here — every entry parsed by
    this function is a real, scoreable arm; the per-language `control:
    untuned_base` string is parsed separately and never produces a
    `Candidate`."""
    candidate_id = raw.get("id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise BakeoffConfigError(f"{path}: {language} arm missing a non-empty 'id'")

    hf_repo = raw.get("hf_repo")
    if not isinstance(hf_repo, str) or not hf_repo:
        raise BakeoffConfigError(f"{path}: {language}/{candidate_id} missing a non-empty 'hf_repo'")

    license_raw = raw.get("license")
    if license_raw is None:
        raise BakeoffConfigError(
            f"{path}: {language}/{candidate_id}: hf_repo is set but license is missing — "
            "no candidate enters the bake-off until its license is confirmed (tech-spec v2 §3.2)"
        )
    if not isinstance(license_raw, str) or not license_raw:
        raise BakeoffConfigError(
            f"{path}: {language}/{candidate_id}: license must be a non-empty string, "
            f"got {type(license_raw).__name__}"
        )

    optional_raw = raw.get("optional", False)
    if not isinstance(optional_raw, bool):
        raise BakeoffConfigError(f"{path}: {language}/{candidate_id}.optional must be a bool")

    instruct_raw = raw.get("instruct", defaults.instruct)
    if not isinstance(instruct_raw, bool):
        raise BakeoffConfigError(f"{path}: {language}/{candidate_id}.instruct must be a bool")

    alt_raw = raw.get("alt")
    if alt_raw is not None and not isinstance(alt_raw, str):
        raise BakeoffConfigError(f"{path}: {language}/{candidate_id}.alt must be a string or null")

    base_raw = raw.get("base")
    if base_raw is not None and not isinstance(base_raw, str):
        raise BakeoffConfigError(f"{path}: {language}/{candidate_id}.base must be a string or null")

    lang_token_raw = raw.get("lang_token")
    if lang_token_raw is not None and not isinstance(lang_token_raw, str):
        raise BakeoffConfigError(f"{path}: {language}/{candidate_id}.lang_token must be a string or null")

    note = raw.get("note")
    if note is not None and not isinstance(note, str):
        raise BakeoffConfigError(f"{path}: {language}/{candidate_id}.note must be a string or null")

    # License allowlist enforcement (issue #27 "License allowlist" proposed
    # resolution, promotes backlog 0015). A `[VERIFY]`-tagged license is
    # accepted as a documented placeholder but flagged as a loader warning,
    # distinct from a hard rejection — an arm marked `optional: true` is a
    # pure-baseline-only role and is exempt from the hard-reject path (it
    # never becomes a release-eligible adapter), matching this ticket's
    # "any release-eligible arm (i.e. any arm not marked optional: true in a
    # pure-baseline-only role)" wording.
    normalized_license = license_raw.strip().lower()
    if normalized_license == "[verify]":
        warnings.append(
            f"{language}/{candidate_id}: license is '[VERIFY]' — accepted as a documented "
            "placeholder, must be confirmed before any release-class decision (tech-spec v2 §3.2)"
        )
    elif not optional_raw and normalized_license not in allowlist:
        raise BakeoffConfigError(
            f"{language}/{candidate_id}: license {license_raw!r} is not in the release "
            f"allowlist {sorted(allowlist)} (defaults.release_licenses)"
        )

    return Candidate(
        id=candidate_id,
        hf_repo=hf_repo,
        license=license_raw,
        size_b=_parse_optional_int(raw, "size_b", path, f"{language}/{candidate_id}"),
        size_m=_parse_optional_int(raw, "size_m", path, f"{language}/{candidate_id}"),
        instruct=instruct_raw,
        optional=optional_raw,
        alt=alt_raw,
        base=base_raw,
        lang_token=lang_token_raw,
        note=note,
    )


def _parse_control(raw: dict[str, object], language: str, path: Path) -> str:
    control_raw = raw.get("control")
    if not isinstance(control_raw, str) or not control_raw:
        raise BakeoffConfigError(
            f"{path}: {language}.control must be a non-empty string naming the control "
            "policy (tech-spec v2 §3.2; e.g. 'untuned_base')"
        )
    return control_raw


def load_bakeoff_config(path: Path) -> LoadedBakeoffConfig:
    """Parse and validate `bakeoff_candidates.yml` (contract v2, tech-spec v2
    §3.2 config surface). Raises `BakeoffConfigError` on: a missing/invalid
    `schema_version` or `defaults` block; a `frc` section without a flat
    `arms:` list; a `lou` section without both `generative_arms:` (2 entries)
    and `mt_arms:` (3 entries); any arm missing `id`/`hf_repo`/`license`; or
    any release-eligible arm's license outside `defaults.release_licenses`.
    Returns the parsed `BakeoffConfig` plus a `warnings` list carrying any
    `[VERIFY]`-tagged license (issue #27)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw, dict):
        raise BakeoffConfigError(f"{path}: top level must be a mapping")

    schema_version = raw.get("schema_version")
    if schema_version != "2.0.0":
        raise BakeoffConfigError(
            f"{path}: schema_version must be '2.0.0', got {schema_version!r} (contract v2, MIG-01c)"
        )

    defaults_raw = raw.get("defaults")
    if not isinstance(defaults_raw, dict):
        raise BakeoffConfigError(f"{path}: top-level 'defaults' block is required")
    defaults = _parse_defaults(defaults_raw, path)
    allowlist = frozenset(license_id.strip().lower() for license_id in defaults.release_licenses)

    unsupported = set(raw.keys()) - _SUPPORTED_LANGUAGES - {"schema_version", "defaults"}
    if unsupported:
        raise BakeoffConfigError(f"{path}: unsupported top-level key(s) {sorted(unsupported)}")

    frc_raw = raw.get("frc")
    if not isinstance(frc_raw, dict):
        raise BakeoffConfigError(f"{path}: top-level 'frc' section is required")

    warnings: list[str] = []

    frc_control = _parse_control(frc_raw, "frc", path)
    frc_arms_raw = frc_raw.get("arms")
    if not isinstance(frc_arms_raw, list) or not frc_arms_raw:
        raise BakeoffConfigError(f"{path}: frc.arms must be a non-empty list")
    frc_arms = tuple(
        _parse_arm(
            arm,
            language="frc",
            path=path,
            defaults=defaults,
            allowlist=allowlist,
            warnings=warnings,
        )
        for arm in frc_arms_raw
    )
    frc_config = FrcLanguageConfig(control=frc_control, arms=frc_arms)

    lou_raw = raw.get("lou")
    if not isinstance(lou_raw, dict):
        raise BakeoffConfigError(f"{path}: top-level 'lou' section is required")

    lou_control = _parse_control(lou_raw, "lou", path)

    generative_arms_raw = lou_raw.get("generative_arms")
    if not isinstance(generative_arms_raw, list) or not generative_arms_raw:
        raise BakeoffConfigError(f"{path}: lou.generative_arms must be a non-empty list")
    generative_arms = tuple(
        _parse_arm(
            arm,
            language="lou",
            path=path,
            defaults=defaults,
            allowlist=allowlist,
            warnings=warnings,
        )
        for arm in generative_arms_raw
    )

    mt_arms_raw = lou_raw.get("mt_arms")
    if not isinstance(mt_arms_raw, list) or not mt_arms_raw:
        raise BakeoffConfigError(f"{path}: lou.mt_arms must be a non-empty list")
    mt_arms = tuple(
        _parse_arm(
            arm,
            language="lou",
            path=path,
            defaults=defaults,
            allowlist=allowlist,
            warnings=warnings,
        )
        for arm in mt_arms_raw
    )

    prerequisite_raw = lou_raw.get("prerequisite")
    if not isinstance(prerequisite_raw, str) or not prerequisite_raw:
        raise BakeoffConfigError(f"{path}: lou.prerequisite must be a non-empty string (Open Q15)")

    decision_rule_raw = lou_raw.get("decision_rule")
    if not isinstance(decision_rule_raw, str) or not decision_rule_raw:
        raise BakeoffConfigError(f"{path}: lou.decision_rule must be a non-empty string (Open Q13)")

    lou_config = LouLanguageConfig(
        control=lou_control,
        generative_arms=generative_arms,
        mt_arms=mt_arms,
        prerequisite=prerequisite_raw,
        decision_rule=decision_rule_raw,
    )

    config = BakeoffConfig(
        schema_version=schema_version,
        defaults=defaults,
        frc=frc_config,
        lou=lou_config,
    )
    return LoadedBakeoffConfig(config=config, warnings=warnings)


@dataclass(frozen=True, slots=True)
class TrainedArtifact:
    """What `fine_tune` returns: an opaque, typed handle to a trained
    candidate (R5 — two real adapters, e.g. in-process vs subprocess,
    previously had no shared type to agree on). `adapter_ref` is
    backend-defined (a path, an adapter directory, a run id understood by
    the training backend) — this module does not interpret it, only threads
    it between `fine_tune`, `score`, and `run_red_team` unchanged.

    `hyperparameters_digest` is a backend-computed stable digest (e.g. a
    hash) of the hyperparameters actually used, so `score`/`run_red_team`
    implementations and the model card can confirm which run produced this
    artifact without this module needing to know the concrete
    hyperparameters type. `seed` records which of `run_bakeoff`'s `seeds`
    produced this artifact (issue #27 "seeds threading").
    """

    candidate_id: str
    adapter_ref: str
    run_id: str
    hyperparameters_digest: str
    seed: int


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """`score`'s return value: which metric was used, its value, and the
    direction that counts as better — replacing the pre-#14 bare `float`
    whose aggregation rule and comparison direction were unstated and
    unrecorded (R5). `run_bakeoff` uses `higher_is_better` instead of
    assuming `max()` is always correct, and raises if candidates within one
    run disagree on `metric_name`/`higher_is_better` (tech-spec v2 §3.2
    harness responsibility 2: "the same metric set")."""

    metric_name: str
    value: float
    higher_is_better: bool


@dataclass(frozen=True, slots=True)
class SeedAggregatedScore:
    """The aggregation of one candidate's `ScoreResult` values across
    `run_bakeoff`'s `seeds` (issue #27 "seeds threading" proposed
    resolution) — mean and spread (max - min) rather than a single-seed
    `ScoreResult`, so a single-seed run is never silently reported as the
    tech-spec v2 §3.2/§5 required multi-seed result. `metric_name`/
    `higher_is_better` are carried through unchanged from the per-seed
    `ScoreResult` values (already validated identical across seeds by
    `run_bakeoff`, the same way they are validated identical across
    candidates)."""

    metric_name: str
    mean: float
    spread: float
    higher_is_better: bool
    per_seed: tuple[ScoreResult, ...]


# --- Red-team verdict (ADR 0008; tech-spec v2 §6.3) --------------------------

GateClass = Literal[0, 1, 2, 3]


@dataclass(frozen=True, slots=True)
class RedTeamCellResult:
    """One diff-catalog cell's red-team outcome (tech-spec v2 §6.3 probe
    schema / severity table): base and tuned failure rates on the same
    locked probes, a Wilson 95% CI on the tuned rate, a paired McNemar
    p-value (`None` when n < 30 and the point-estimate rule is used
    instead, per ADR 0008 decision 1(a)), and the resulting `gate_class`
    (the diff-catalog cell's own severity, §4) versus `class_assigned` (this
    run's actual outcome for the cell, which may differ from `gate_class`
    when e.g. n was too small to reach statistical significance — ADR 0008
    decision 1 evaluates (a)/(b)/(c) against the observed data, not the
    catalog's a-priori severity alone)."""

    cell_id: str
    base_rate: float
    tuned_rate: float
    wilson_95: tuple[float, float]
    mcnemar_p: float | None
    gate_class: GateClass
    class_assigned: GateClass


@dataclass(frozen=True, slots=True)
class RedTeamVerdict:
    """`run_red_team`'s return value (ADR 0008; tech-spec v2 §6.3): replaces
    `RedTeamReport`/`ProbeVerdict` (issue #14's v1 "zero critical failures"
    rule, unpassable at 320 probes per R0 Part B) with the class-based,
    baseline-calibrated gate. `worst_class` is the minimum `class_assigned`
    across `cells` (Class 0 is worst per the severity table), and
    `disqualified` is derived from it — `worst_class <= 1` — never set
    directly, so the disqualification precedence rule has exactly one
    source, matching the superseded `RedTeamReport.disqualified`'s own
    single-source design."""

    probe_set_version: str
    cells: dict[str, RedTeamCellResult]

    @property
    def worst_class(self) -> int:
        if not self.cells:
            raise BakeoffError(
                "RedTeamVerdict.worst_class is undefined for a verdict with no cells — "
                "a real red-team run always scores at least one cell"
            )
        return min(cell.class_assigned for cell in self.cells.values())

    @property
    def disqualified(self) -> bool:
        return self.worst_class <= 1


def derive_model_release_class(
    verdict: RedTeamVerdict,
    *,
    license_lineage_clear: bool,
) -> ModelReleaseClass:
    """Pure derivation of a candidate's `ModelReleaseClass` from its
    `RedTeamVerdict` (ADR 0008 decision 4; issue #27 "CandidateResult.
    release_class" proposed resolution). Callable independently of
    `run_bakeoff` so `src/eval/`'s `AcceptanceReport.release_class`
    (MIG-01e) can reuse it without a cross-sibling import of `bakeoff`
    (issue #27 story 10).

    `release_ready` only when `worst_class >= 2` (no Class 0/1 finding) AND
    `license_lineage_clear` (an NC license lineage forces `research_only`
    even at `worst_class >= 2`, per ADR 0008 decision 4's release-class
    definitions); `research_only` otherwise. This function never returns
    `internal_only` or `withdrawn` — both are assigned by other governance
    paths (ADR 0008 decision 4: `internal_only` from closed-tier training
    data under ADR 0007; `withdrawn` from a revoked voice/withdrawn item),
    not from a red-team verdict alone.
    """
    if verdict.worst_class >= 2 and license_lineage_clear:
        return "release_ready"
    return "research_only"


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """One candidate's outcome: seed-aggregated score, red-team verdict, the
    untuned base's own score (tech-spec v2 §3.2 harness responsibility 3),
    derived disqualification, and derived model release class.
    `disqualified` is always True when `red_team_verdict.disqualified` is
    True, regardless of score (tech-spec v2 §3.2 harness responsibility 3's
    precedence rule). `score` is typed as optional for a future ticket that
    adds failure handling around the injected callables (out of scope here
    — this skeleton propagates any exception from
    fine_tune/score/score_untuned_base/run_red_team directly, it does not
    catch and convert it to a None score)."""

    candidate_id: str
    untuned_base_score: ScoreResult | None
    score: SeedAggregatedScore | None
    red_team_verdict: RedTeamVerdict
    disqualified: bool
    release_class: ModelReleaseClass
    raw_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class BakeoffRunResult:
    """Every candidate's outcome (not just the winner — tech-spec v2 §3.2
    harness responsibility 4), plus the derived winner. winner_candidate_id
    is None, never an arbitrary pick, if every candidate is disqualified."""

    language: TargetLanguage
    results: tuple[CandidateResult, ...]
    winner_candidate_id: str | None


def run_bakeoff(
    language: TargetLanguage,
    config: BakeoffConfig,
    *,
    hyperparameters: H,
    split_id: str,
    seeds: Sequence[int],
    fine_tune: Callable[[Candidate, H, str, int], TrainedArtifact],
    score: Callable[[TrainedArtifact, TargetLanguage], tuple[ScoreResult, dict[str, float]]],
    score_untuned_base: Callable[[Candidate, TargetLanguage], ScoreResult],
    run_red_team: Callable[[TrainedArtifact, TargetLanguage], RedTeamVerdict],
    license_lineage_clear: Callable[[Candidate], bool] = lambda candidate: True,
) -> BakeoffRunResult:
    """Fine-tune, score, and red-team every candidate for `language`
    identically except for the base model itself (tech-spec v2 §3.2 harness
    responsibility 1 — enforced structurally: every candidate goes through
    the same stages in the same order, every `fine_tune` call receives the
    identical `hyperparameters`/`split_id`, and every candidate is fine-tuned
    once per entry of the identical `seeds` sequence). For each candidate,
    `score_untuned_base` runs once, before that candidate's per-seed
    fine_tune/score/run_red_team sequence (tech-spec v2 §3.2 harness
    responsibility 3). See module docstring: the seams are for future
    tickets, not implemented here.

    Raises `BakeoffError` if `language` is not a key of `config`, if `seeds`
    is empty, or if candidates disagree on `metric_name`/`higher_is_better`
    (checked across both `score` and `score_untuned_base` results).
    """
    arms = config.arms_for(language)
    if not arms:
        raise BakeoffError(
            f"no bake-off arms configured for language={language!r}"
        )
    if language not in _SUPPORTED_LANGUAGES:
        raise BakeoffError(f"unsupported language={language!r}; expected one of {sorted(_SUPPORTED_LANGUAGES)}")

    seed_list = list(seeds)
    if not seed_list:
        raise BakeoffError("seeds must be non-empty (tech-spec v2 §3.2 harness responsibility 1)")

    results: list[CandidateResult] = []
    metric_name: str | None = None
    higher_is_better: bool | None = None

    def _check_metric(candidate_score: ScoreResult, candidate_id: str) -> None:
        nonlocal metric_name, higher_is_better
        if metric_name is None:
            metric_name = candidate_score.metric_name
            higher_is_better = candidate_score.higher_is_better
        elif (
            candidate_score.metric_name != metric_name
            or candidate_score.higher_is_better != higher_is_better
        ):
            raise BakeoffError(
                f"{language}: candidates disagree on scoring metric — "
                f"expected ({metric_name!r}, higher_is_better={higher_is_better}), "
                f"got ({candidate_score.metric_name!r}, "
                f"higher_is_better={candidate_score.higher_is_better}) from {candidate_id!r}"
            )

    for candidate in arms:
        untuned_base_score = score_untuned_base(candidate, language)
        _check_metric(untuned_base_score, candidate.id)

        per_seed_scores: list[ScoreResult] = []
        raw_metrics: dict[str, float] = {}
        last_artifact: TrainedArtifact | None = None

        for seed in seed_list:
            artifact = fine_tune(candidate, hyperparameters, split_id, seed)
            candidate_score, seed_raw_metrics = score(artifact, language)
            _check_metric(candidate_score, candidate.id)
            per_seed_scores.append(candidate_score)
            raw_metrics.update(seed_raw_metrics)
            last_artifact = artifact

        assert last_artifact is not None  # seed_list is non-empty, checked above

        # The red-team gate is scored once per candidate (not once per seed)
        # — ADR 0008 does not require a per-seed gate run, and
        # `run_red_team`'s cost (a full locked-probe pass) is not meant to
        # multiply by seed count. The last seed's artifact is used; any
        # future per-seed red-team requirement is out of scope here.
        red_team_verdict = run_red_team(last_artifact, language)

        aggregated = SeedAggregatedScore(
            metric_name=metric_name if metric_name is not None else per_seed_scores[0].metric_name,
            mean=sum(s.value for s in per_seed_scores) / len(per_seed_scores),
            spread=max(s.value for s in per_seed_scores) - min(s.value for s in per_seed_scores),
            higher_is_better=higher_is_better if higher_is_better is not None else per_seed_scores[0].higher_is_better,
            per_seed=tuple(per_seed_scores),
        )

        release_class = derive_model_release_class(
            red_team_verdict,
            license_lineage_clear=license_lineage_clear(candidate),
        )

        results.append(
            CandidateResult(
                candidate_id=candidate.id,
                untuned_base_score=untuned_base_score,
                score=aggregated,
                red_team_verdict=red_team_verdict,
                disqualified=red_team_verdict.disqualified,
                release_class=release_class,
                raw_metrics=raw_metrics,
            )
        )

    # Tie-break: on an exact score tie, the higher candidate_id string wins
    # regardless of higher_is_better — this is an arbitrary but deterministic
    # choice, not derived from the metric direction (unchanged from the
    # pre-#27 rule verbatim; issue #27 "Winner rule unchanged"). Implemented
    # by best-score-first selection (direction-aware) then a max() over
    # candidate_id among everyone tied with the best score, rather than raw
    # tuple comparison (which would flip the tie-break direction when
    # higher_is_better=False).
    eligible = [
        (r.candidate_id, r.score.mean) for r in results if not r.disqualified and r.score is not None
    ]
    if eligible:
        best_value = (max if higher_is_better else min)(value for _, value in eligible)
        winner_id = max(cid for cid, value in eligible if value == best_value)
    else:
        winner_id = None

    return BakeoffRunResult(language=language, results=tuple(results), winner_candidate_id=winner_id)
