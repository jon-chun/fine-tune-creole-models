"""Base-model bake-off: config loader + orchestration skeleton.

tech-spec §3.2: base-model selection is never assumed, always an empirical
bake-off. This module builds two real pieces (the config loader and the
result-aggregation orchestration in run_bakeoff) around three stub-shaped
seams — `fine_tune`, `score`, `run_red_team` — that a later ticket fills in
with real implementations:

- `fine_tune`: the LoRA/QLoRA/DoRA training driver (tech-spec §5). Not built
  here. Injected as a callable so run_bakeoff never imports a concrete
  trainer.
- `score`: the Layer-B gold-set scoring harness (tech-spec §6). Not built
  here — no gold set exists yet either.
- `run_red_team`: the conflation red-team suite (tech-spec §6.3). Not built
  here.

Calling run_bakeoff() with placeholder/test-double callables exercises the
orchestration logic only — candidate iteration, disqualification
precedence, winner selection — never a real bake-off result. This module
ships no training, scoring, or red-team logic of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from data_contract import LanguageTag

_SUPPORTED_LANGUAGES = frozenset({"frc", "lou"})


class BakeoffConfigError(ValueError):
    """Raised when bakeoff_candidates.yml is malformed or fails validation."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One bake-off candidate: either a real base model (hf_repo + license
    both set) or the lower-transfer control (base: null, no hf_repo,
    exempt from the license check since it has nothing to license)."""

    id: str
    hf_repo: str | None
    license: str | None
    base: None
    note: str | None


BakeoffConfig = dict[LanguageTag, list[Candidate]]


def _parse_candidate(raw: dict[str, object], language: str) -> Candidate:
    candidate_id = raw.get("id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise BakeoffConfigError(f"{language}: candidate missing a non-empty 'id'")

    hf_repo = raw.get("hf_repo")
    license_ = raw.get("license")
    note = raw.get("note")

    if hf_repo is not None and not license_:
        raise BakeoffConfigError(
            f"{language}/{candidate_id}: hf_repo is set but license is missing — "
            "no candidate enters the bake-off until its license is confirmed (tech-spec §3.2)"
        )

    return Candidate(
        id=candidate_id,
        hf_repo=hf_repo if isinstance(hf_repo, str) else None,
        license=license_ if isinstance(license_, str) else None,
        base=None,
        note=note if isinstance(note, str) else None,
    )


def load_bakeoff_config(path: Path) -> BakeoffConfig:
    """Parse and validate bakeoff_candidates.yml. Raises BakeoffConfigError
    on any language key outside {frc, lou}, or any hf_repo-bearing candidate
    missing a license."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw, dict):
        raise BakeoffConfigError(f"{path}: top level must be a mapping of language -> candidates")

    unsupported = set(raw.keys()) - _SUPPORTED_LANGUAGES
    if unsupported:
        raise BakeoffConfigError(f"{path}: unsupported language key(s) {sorted(unsupported)}")

    config: BakeoffConfig = {}
    for language, section in raw.items():
        raw_candidates = section.get("candidates", []) if isinstance(section, dict) else []
        config[language] = [_parse_candidate(c, language) for c in raw_candidates]

    return config


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """One candidate's outcome: score, red-team verdict, and derived
    disqualification — disqualified is always True when red_team_passed is
    False, regardless of score (tech-spec §3.2 harness responsibility #3's
    precedence rule). `score` is typed as optional for a future ticket that
    adds failure handling around the injected callables (out of scope
    here — this skeleton propagates any exception from fine_tune/score/
    run_red_team directly, it does not catch and convert it to a None
    score)."""

    candidate_id: str
    score: float | None
    red_team_passed: bool
    disqualified: bool
    raw_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class BakeoffRunResult:
    """Every candidate's outcome (not just the winner — tech-spec §3.2
    harness responsibility #4), plus the derived winner. winner_candidate_id
    is None, never an arbitrary pick, if every candidate is disqualified."""

    language: LanguageTag
    results: tuple[CandidateResult, ...]
    winner_candidate_id: str | None


def run_bakeoff(
    language: LanguageTag,
    config: BakeoffConfig,
    *,
    fine_tune: Callable[[Candidate], object],
    score: Callable[[object, LanguageTag], tuple[float, dict[str, float]]],
    run_red_team: Callable[[object, LanguageTag], bool],
) -> BakeoffRunResult:
    """Fine-tune, score, and red-team every candidate for `language`
    identically except for the base model itself (tech-spec §3.2 harness
    responsibility #1 — enforced structurally: every candidate goes through
    the same three stages in the same order). See module docstring: the
    three callables are seams for future tickets, not implemented here.
    """
    results: list[CandidateResult] = []

    for candidate in config.get(language, []):
        artifact = fine_tune(candidate)
        candidate_score, raw_metrics = score(artifact, language)
        red_team_passed = run_red_team(artifact, language)

        results.append(
            CandidateResult(
                candidate_id=candidate.id,
                score=candidate_score,
                red_team_passed=red_team_passed,
                disqualified=not red_team_passed,
                raw_metrics=raw_metrics,
            )
        )

    # Tie-break: on an exact score tie, the higher candidate_id string wins
    # (tuple comparison falls through to the second element). This is an
    # arbitrary but deterministic choice — ties are not expected to be
    # common in practice, and no tech-spec section specifies a tie-break
    # rule, so any deterministic rule is defensible; documented here so a
    # future reader isn't surprised by it.
    eligible_scores = [(r.score, r.candidate_id) for r in results if not r.disqualified and r.score is not None]
    winner_id = max(eligible_scores)[1] if eligible_scores else None

    return BakeoffRunResult(language=language, results=tuple(results), winner_candidate_id=winner_id)
