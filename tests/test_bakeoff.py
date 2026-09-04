"""Tests for src/bakeoff/ — config loader and run_bakeoff() orchestration
(contract v2, ADR 0008, ADR 0010; issue #27 / MIG-01c).

Tests only external behavior: loading config fixtures / calling
run_bakeoff() with injected test-double callables, asserting on the
returned types. Never asserts on internal call sequencing beyond call
counts or explicit ordering claims the ticket makes (an explicit
orchestration contract, not an implementation detail).
"""

from pathlib import Path

import pytest

from bakeoff import (
    BakeoffConfig,
    BakeoffConfigError,
    BakeoffDefaults,
    BakeoffError,
    Candidate,
    FrcLanguageConfig,
    GateClass,
    LouLanguageConfig,
    RedTeamCellResult,
    RedTeamVerdict,
    ScoreResult,
    SeedAggregatedScore,
    TrainedArtifact,
    derive_model_release_class,
    load_bakeoff_config,
    run_bakeoff,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "bakeoff_candidates.yml"

# A stand-in hyperparameters value. bakeoff must not import train.Hyperparameters
# (no cross-sibling import — issue #14, restated issue #27), so tests use a
# plain dict/mapping to exercise the generic H TypeVar without pulling in the
# real type.
_HYPERPARAMETERS = {"rank": 16, "alpha": 32, "learning_rate": 2e-4}
_SPLIT_ID = "split-2026-09-02"
_SEEDS = (1, 2, 3)


# --- Config loader: the real committed file ----------------------------------


def test_real_config_frc_arms_match_tech_spec_v2() -> None:
    loaded = load_bakeoff_config(REAL_CONFIG_PATH)
    ids = {c.id for c in loaded.config.frc.arms}
    assert ids == {
        "A-lucie-7b-instruct",
        "B-ministral-3-8b",
        "C-salamandra-7b",
        "D-smollm3-3b",
        "F-claire-7b-apache",
    }


def test_real_config_lou_generative_and_mt_arms_match_tech_spec_v2() -> None:
    loaded = load_bakeoff_config(REAL_CONFIG_PATH)
    assert len(loaded.config.lou.generative_arms) == 2
    assert len(loaded.config.lou.mt_arms) == 3
    mt_ids = {c.hf_repo for c in loaded.config.lou.mt_arms}
    assert "jhu-clsp/kreyol-mt" in mt_ids
    assert "facebook/mbart-large-50-many-to-many-mmt" in mt_ids


def test_control_untuned_base_is_not_a_candidate_row() -> None:
    loaded = load_bakeoff_config(REAL_CONFIG_PATH)
    assert loaded.config.frc.control == "untuned_base"
    assert loaded.config.lou.control == "untuned_base"
    frc_ids = {c.id for c in loaded.config.frc.arms}
    lou_ids = {c.id for c in loaded.config.lou.generative_arms + loaded.config.lou.mt_arms}
    assert "control-low-transfer" not in frc_ids
    assert "control-low-transfer" not in lou_ids


# --- Config loader: license allowlist (issue #27, promotes backlog 0015) ----


_MINIMAL_DEFAULTS = (
    'schema_version: "2.0.0"\n'
    "defaults:\n"
    "  instruct: required\n"
    "  seeds: 3\n"
    "  forgetting_axis: required\n"
    "  release_licenses: [apache-2.0, mit]\n"
)

_MINIMAL_LOU = (
    "lou:\n"
    "  control: untuned_base\n"
    "  generative_arms:\n"
    '    - {id: g1, hf_repo: "org/g1", license: apache-2.0}\n'
    '    - {id: g2, hf_repo: "org/g2", license: apache-2.0}\n'
    "  mt_arms:\n"
    '    - {id: m1, hf_repo: "org/m1", license: mit}\n'
    '    - {id: m2, hf_repo: "org/m2", license: mit}\n'
    '    - {id: m3, hf_repo: "org/m3", license: mit}\n'
    '  prerequisite: "some prerequisite"\n'
    '  decision_rule: "some rule"\n'
)


def test_license_outside_allowlist_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: sketchy, hf_repo: "someorg/some-model", license: gpl-3.0}\n'
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError, match="sketchy"):
        load_bakeoff_config(bad_config)


def test_verify_tagged_license_loads_with_warning(tmp_path: Path) -> None:
    ok_config = tmp_path / "ok.yml"
    ok_config.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: pending, hf_repo: "someorg/some-model", license: "[VERIFY]"}\n'
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    loaded = load_bakeoff_config(ok_config)
    assert loaded.config.frc.arms[0].id == "pending"
    assert any("pending" in w for w in loaded.warnings)


def test_optional_arm_with_unlisted_license_is_not_hard_rejected(tmp_path: Path) -> None:
    # An arm marked optional: true in a pure-baseline-only role is exempt
    # from the hard-reject path (issue #27: "any arm not marked optional:
    # true in a pure-baseline-only role").
    ok_config = tmp_path / "ok.yml"
    ok_config.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: a, hf_repo: "someorg/a", license: apache-2.0}\n'
        '    - {id: baseline-only, hf_repo: "someorg/b", license: gpl-3.0, optional: true}\n'
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    loaded = load_bakeoff_config(ok_config)
    ids = {c.id for c in loaded.config.frc.arms}
    assert "baseline-only" in ids


def test_language_key_outside_frc_lou_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: a, hf_repo: "org/a", license: apache-2.0}\n'
        + _MINIMAL_LOU
        + "hat:\n"
        "  arms:\n"
        '    - {id: x, hf_repo: "org/x", license: apache-2.0}\n',
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError):
        load_bakeoff_config(bad_config)


def test_arm_missing_hf_repo_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        "    - {id: mystery}\n"
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError):
        load_bakeoff_config(bad_config)


def test_arm_with_hf_repo_and_no_license_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: sketchy, hf_repo: "someorg/some-model"}\n'
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError):
        load_bakeoff_config(bad_config)


def test_candidate_with_non_string_license_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: sketchy, hf_repo: "someorg/some-model", license: 42}\n'
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError):
        load_bakeoff_config(bad_config)


def test_missing_schema_version_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        "defaults:\n"
        "  instruct: required\n"
        "  seeds: 3\n"
        "  forgetting_axis: required\n"
        "  release_licenses: [apache-2.0]\n"
        "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: a, hf_repo: "org/a", license: apache-2.0}\n'
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError):
        load_bakeoff_config(bad_config)


def test_lou_missing_mt_arms_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: a, hf_repo: "org/a", license: apache-2.0}\n'
        "lou:\n"
        "  control: untuned_base\n"
        "  generative_arms:\n"
        '    - {id: g1, hf_repo: "org/g1", license: apache-2.0}\n'
        '    - {id: g2, hf_repo: "org/g2", license: apache-2.0}\n'
        '  prerequisite: "x"\n'
        '  decision_rule: "y"\n',
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError):
        load_bakeoff_config(bad_config)


def test_arm_size_and_instruct_fields_are_typed(tmp_path: Path) -> None:
    config_path = tmp_path / "sized.yml"
    config_path.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: a, hf_repo: "org/a", license: apache-2.0, size_b: 7, instruct: false, optional: true, alt: "org/a-alt"}\n'
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    loaded = load_bakeoff_config(config_path)
    arm = loaded.config.frc.arms[0]
    assert arm.size_b == 7
    assert arm.instruct is False
    assert arm.optional is True
    assert arm.alt == "org/a-alt"


def test_arm_instruct_defaults_from_defaults_block(tmp_path: Path) -> None:
    config_path = tmp_path / "defaulted.yml"
    config_path.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: a, hf_repo: "org/a", license: apache-2.0}\n'
        + _MINIMAL_LOU,
        encoding="utf-8",
    )
    loaded = load_bakeoff_config(config_path)
    assert loaded.config.frc.arms[0].instruct is True


def test_mt_arm_carries_size_m_base_and_lang_token(tmp_path: Path) -> None:
    config_path = tmp_path / "mt.yml"
    config_path.write_text(
        _MINIMAL_DEFAULTS
        + "frc:\n"
        "  control: untuned_base\n"
        "  arms:\n"
        '    - {id: a, hf_repo: "org/a", license: apache-2.0}\n'
        "lou:\n"
        "  control: untuned_base\n"
        "  generative_arms:\n"
        '    - {id: g1, hf_repo: "org/g1", license: apache-2.0}\n'
        '    - {id: g2, hf_repo: "org/g2", license: apache-2.0}\n'
        "  mt_arms:\n"
        '    - {id: m1, hf_repo: "jhu-clsp/kreyol-mt", license: mit, size_m: 611, base: "facebook/mbart-large-50-many-to-many-mmt", lang_token: "<2lou>"}\n'
        '    - {id: m2, hf_repo: "org/m2", license: mit}\n'
        '    - {id: m3, hf_repo: "org/m3", license: mit}\n'
        '  prerequisite: "x"\n'
        '  decision_rule: "y"\n',
        encoding="utf-8",
    )
    loaded = load_bakeoff_config(config_path)
    kreyol = next(a for a in loaded.config.lou.mt_arms if a.id == "m1")
    assert kreyol.size_m == 611
    assert kreyol.base == "facebook/mbart-large-50-many-to-many-mmt"
    assert kreyol.lang_token == "<2lou>"


# --- run_bakeoff() orchestration ---------------------------------------------


def _candidate(id_: str, *, optional: bool = False) -> Candidate:
    return Candidate(
        id=id_,
        hf_repo=f"org/{id_}",
        license="apache-2.0",
        size_b=None,
        size_m=None,
        instruct=True,
        optional=optional,
        alt=None,
        base=None,
        lang_token=None,
        note=None,
    )


def _config_with(candidates: list[Candidate]) -> BakeoffConfig:
    defaults = BakeoffDefaults(
        instruct=True,
        seeds=3,
        forgetting_axis="required",
        release_licenses=("apache-2.0", "mit"),
        nf4_min_size_b=20,
    )
    frc = FrcLanguageConfig(control="untuned_base", arms=tuple(candidates))
    lou = LouLanguageConfig(
        control="untuned_base",
        generative_arms=(_candidate("lou-g1"), _candidate("lou-g2")),
        mt_arms=(_candidate("lou-m1"), _candidate("lou-m2"), _candidate("lou-m3")),
        prerequisite="x",
        decision_rule="y",
    )
    return BakeoffConfig(schema_version="2.0.0", defaults=defaults, frc=frc, lou=lou)


def _artifact(candidate: Candidate, seed: int) -> TrainedArtifact:
    return TrainedArtifact(
        candidate_id=candidate.id,
        adapter_ref=f"adapter://{candidate.id}/{seed}",
        run_id=f"run-{candidate.id}-{seed}",
        hyperparameters_digest="digest-fixed",
        seed=seed,
    )


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


def _passing_red_team() -> RedTeamVerdict:
    return RedTeamVerdict(probe_set_version="v2026-09-03", cells={"VERB-001": _cell(class_assigned=3)})


def _failing_red_team() -> RedTeamVerdict:
    return RedTeamVerdict(probe_set_version="v2026-09-03", cells={"VERB-001": _cell(class_assigned=1)})


def _untuned_base_score(metric: str = "gold_accuracy", value: float = 0.4, higher_is_better: bool = True) -> ScoreResult:
    return ScoreResult(metric, value, higher_is_better=higher_is_better)


def test_highest_scoring_non_disqualified_candidate_wins() -> None:
    candidates = [_candidate("low"), _candidate("high")]
    scores = {"low": 0.5, "high": 0.9}

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", scores[artifact.candidate_id], higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert result.winner_candidate_id == "high"


def test_lower_is_better_metric_picks_lowest_score() -> None:
    candidates = [_candidate("low"), _candidate("high")]
    scores = {"low": 5.0, "high": 20.0}

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("perplexity", scores[artifact.candidate_id], higher_is_better=False), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score("perplexity", 30.0, higher_is_better=False)

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert result.winner_candidate_id == "low"


def test_every_candidate_appears_in_results_not_just_winner() -> None:
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert {r.candidate_id for r in result.results} == {"a", "b", "c"}


def test_all_disqualified_yields_no_winner_not_arbitrary_pick() -> None:
    candidates = [_candidate("a"), _candidate("b")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.9, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _failing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert result.winner_candidate_id is None
    assert all(r.disqualified for r in result.results)


def test_each_stage_called_expected_number_of_times() -> None:
    candidates = [_candidate("a"), _candidate("b")]
    call_counts = {"fine_tune": 0, "score": 0, "run_red_team": 0, "score_untuned_base": 0}

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        call_counts["fine_tune"] += 1
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        call_counts["score"] += 1
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        call_counts["score_untuned_base"] += 1
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        call_counts["run_red_team"] += 1
        return _passing_red_team()

    run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    # 2 candidates x 3 seeds each for fine_tune/score; 1 score_untuned_base
    # and 1 run_red_team per candidate (not per seed — see run_bakeoff
    # docstring).
    assert call_counts == {"fine_tune": 6, "score": 6, "run_red_team": 2, "score_untuned_base": 2}


def test_run_bakeoff_is_deterministic_given_deterministic_callables() -> None:
    candidates = [_candidate("a"), _candidate("b")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        value = 0.7 if artifact.candidate_id == "a" else 0.3
        return ScoreResult("gold_accuracy", value, higher_is_better=True), {"perplexity": 12.3}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    first = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    second = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert first == second


def test_winner_tie_break_unchanged() -> None:
    # Regression test confirming the existing tie-break logic (higher
    # candidate_id string wins on an exact score tie, direction-aware via
    # higher_is_better) is unchanged by this migration (issue #27 "Winner
    # rule unchanged").
    candidates = [_candidate("aaa"), _candidate("zzz")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=True), {}  # exact tie

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert result.winner_candidate_id == "zzz"


def test_raw_metrics_passed_through_unchanged() -> None:
    candidates = [_candidate("a")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=True), {
            "perplexity": 9.1,
            "grammar_probe_accuracy": 0.62,
        }

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert result.results[0].raw_metrics == {"perplexity": 9.1, "grammar_probe_accuracy": 0.62}


def test_missing_language_in_config_raises_instead_of_returning_empty_result() -> None:
    empty_frc = FrcLanguageConfig(control="untuned_base", arms=())
    defaults = BakeoffDefaults(
        instruct=True,
        seeds=3,
        forgetting_axis="required",
        release_licenses=("apache-2.0", "mit"),
        nf4_min_size_b=20,
    )
    lou = LouLanguageConfig(
        control="untuned_base",
        generative_arms=(_candidate("lou-g1"), _candidate("lou-g2")),
        mt_arms=(_candidate("lou-m1"), _candidate("lou-m2"), _candidate("lou-m3")),
        prerequisite="x",
        decision_rule="y",
    )
    config = BakeoffConfig(schema_version="2.0.0", defaults=defaults, frc=empty_frc, lou=lou)

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    with pytest.raises(BakeoffError):
        run_bakeoff(
            "frc",
            config,
            hyperparameters=_HYPERPARAMETERS,
            split_id=_SPLIT_ID,
            seeds=_SEEDS,
            fine_tune=fine_tune,
            score=score,
            score_untuned_base=score_untuned_base,
            run_red_team=run_red_team,
        )


def test_candidates_disagreeing_on_metric_name_raises() -> None:
    candidates = [_candidate("a"), _candidate("b")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        metric = "gold_accuracy" if artifact.candidate_id == "a" else "perplexity"
        return ScoreResult(metric, 0.5, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    with pytest.raises(BakeoffError):
        run_bakeoff(
            "frc",
            _config_with(candidates),
            hyperparameters=_HYPERPARAMETERS,
            split_id=_SPLIT_ID,
            seeds=_SEEDS,
            fine_tune=fine_tune,
            score=score,
            score_untuned_base=score_untuned_base,
            run_red_team=run_red_team,
        )


def test_candidates_disagreeing_on_higher_is_better_raises() -> None:
    candidates = [_candidate("a"), _candidate("b")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        higher_is_better = artifact.candidate_id == "a"
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=higher_is_better), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    with pytest.raises(BakeoffError):
        run_bakeoff(
            "frc",
            _config_with(candidates),
            hyperparameters=_HYPERPARAMETERS,
            split_id=_SPLIT_ID,
            seeds=_SEEDS,
            fine_tune=fine_tune,
            score=score,
            score_untuned_base=score_untuned_base,
            run_red_team=run_red_team,
        )


def test_empty_seeds_raises() -> None:
    candidates = [_candidate("a")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    with pytest.raises(BakeoffError):
        run_bakeoff(
            "frc",
            _config_with(candidates),
            hyperparameters=_HYPERPARAMETERS,
            split_id=_SPLIT_ID,
            seeds=(),
            fine_tune=fine_tune,
            score=score,
            score_untuned_base=score_untuned_base,
            run_red_team=run_red_team,
        )


def test_untuned_base_scored_before_fine_tune_is_called() -> None:
    candidates = [_candidate("a"), _candidate("b")]
    call_order: list[str] = []

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        call_order.append(f"fine_tune:{c.id}:{seed}")
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        call_order.append(f"score_untuned_base:{c.id}")
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )

    # For each candidate, score_untuned_base must precede every fine_tune
    # call for that candidate (tech-spec v2 §3.2 harness responsibility 3).
    a_untuned_index = call_order.index("score_untuned_base:a")
    a_fine_tune_indices = [i for i, c in enumerate(call_order) if c.startswith("fine_tune:a:")]
    assert a_untuned_index < min(a_fine_tune_indices)

    b_untuned_index = call_order.index("score_untuned_base:b")
    b_fine_tune_indices = [i for i, c in enumerate(call_order) if c.startswith("fine_tune:b:")]
    assert b_untuned_index < min(b_fine_tune_indices)


def test_all_seeds_receive_identical_hyperparameters_and_split() -> None:
    # Issue #27 "seeds threading" / issue #14 R5: every (candidate, seed)
    # pair receives the same hyperparameters/split_id.
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    seen: list[tuple[object, str, int]] = []

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        seen.append((hp, split_id, seed))
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.5, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )

    assert len(seen) == 3 * len(_SEEDS)
    assert all(hp == _HYPERPARAMETERS for hp, _, _ in seen)
    assert all(sid == _SPLIT_ID for _, sid, _ in seen)
    assert sorted(seed for _, _, seed in seen) == sorted(list(_SEEDS) * len(candidates))


def test_seed_aggregated_score_reports_mean_and_spread() -> None:
    candidates = [_candidate("a")]
    per_seed_values = {1: 0.6, 2: 0.8, 3: 1.0}

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", per_seed_values[artifact.seed], higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    aggregated = result.results[0].score
    assert isinstance(aggregated, SeedAggregatedScore)
    assert aggregated.mean == pytest.approx(0.8)
    assert aggregated.spread == pytest.approx(0.4)
    assert len(aggregated.per_seed) == 3


def test_untuned_base_score_recorded_on_candidate_result() -> None:
    candidates = [_candidate("a")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.9, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return ScoreResult("gold_accuracy", 0.4, higher_is_better=True)

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert result.results[0].untuned_base_score is not None
    assert result.results[0].untuned_base_score.value == pytest.approx(0.4)


# --- RedTeamVerdict / derive_model_release_class (ADR 0008) ------------------


def test_red_team_verdict_disqualified_true_when_worst_class_le_1() -> None:
    verdict = RedTeamVerdict(
        probe_set_version="v1",
        cells={
            "a": _cell("a", class_assigned=3),
            "b": _cell("b", class_assigned=1),
        },
    )
    assert verdict.worst_class == 1
    assert verdict.disqualified is True


def test_red_team_verdict_not_disqualified_when_worst_class_ge_2() -> None:
    verdict = RedTeamVerdict(
        probe_set_version="v1",
        cells={
            "a": _cell("a", class_assigned=3),
            "b": _cell("b", class_assigned=2),
        },
    )
    assert verdict.worst_class == 2
    assert verdict.disqualified is False


def test_derive_model_release_class_release_ready_when_worst_class_ge_2_and_license_clear() -> None:
    verdict = RedTeamVerdict(probe_set_version="v1", cells={"a": _cell("a", class_assigned=2)})
    assert derive_model_release_class(verdict, license_lineage_clear=True) == "release_ready"


def test_derive_model_release_class_research_only_when_worst_class_le_1() -> None:
    verdict = RedTeamVerdict(probe_set_version="v1", cells={"a": _cell("a", class_assigned=1)})
    assert derive_model_release_class(verdict, license_lineage_clear=True) == "research_only"


def test_derive_model_release_class_research_only_when_license_lineage_not_clear() -> None:
    # NC license lineage forces research_only even at worst_class >= 2
    # (ADR 0008 decision 4).
    verdict = RedTeamVerdict(probe_set_version="v1", cells={"a": _cell("a", class_assigned=3)})
    assert derive_model_release_class(verdict, license_lineage_clear=False) == "research_only"


def test_candidate_result_carries_release_class() -> None:
    candidates = [_candidate("a")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.9, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _passing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert result.results[0].release_class == "release_ready"


def test_candidate_result_release_class_research_only_when_disqualified() -> None:
    candidates = [_candidate("a")]

    def fine_tune(c: Candidate, hp: object, split_id: str, seed: int) -> TrainedArtifact:
        return _artifact(c, seed)

    def score(artifact: TrainedArtifact, language: str) -> tuple[ScoreResult, dict[str, float]]:
        return ScoreResult("gold_accuracy", 0.9, higher_is_better=True), {}

    def score_untuned_base(c: Candidate, language: str) -> ScoreResult:
        return _untuned_base_score()

    def run_red_team(artifact: TrainedArtifact, language: str) -> RedTeamVerdict:
        return _failing_red_team()

    result = run_bakeoff(
        "frc",
        _config_with(candidates),
        hyperparameters=_HYPERPARAMETERS,
        split_id=_SPLIT_ID,
        seeds=_SEEDS,
        fine_tune=fine_tune,
        score=score,
        score_untuned_base=score_untuned_base,
        run_red_team=run_red_team,
    )
    assert result.results[0].release_class == "research_only"
    assert result.results[0].disqualified is True
