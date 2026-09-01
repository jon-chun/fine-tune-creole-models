"""Tests for src/bakeoff/ — config loader and run_bakeoff() orchestration.

Tests only external behavior: loading config fixtures / calling
run_bakeoff() with injected test-double callables, asserting on the
returned types. Never asserts on internal call sequencing beyond call
counts (which are an explicit orchestration contract, not an
implementation detail).
"""

from pathlib import Path

import pytest

from bakeoff import (
    BakeoffConfigError,
    Candidate,
    load_bakeoff_config,
    run_bakeoff,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "bakeoff_candidates.yml"


# --- Config loader: the real committed file ---------------------------------


def test_real_bakeoff_config_loads_with_expected_candidate_counts() -> None:
    config = load_bakeoff_config(REAL_CONFIG_PATH)
    assert len(config["frc"]) == 4
    assert len(config["lou"]) == 2


def test_real_config_frc_candidate_ids_match_tech_spec() -> None:
    config = load_bakeoff_config(REAL_CONFIG_PATH)
    ids = {c.id for c in config["frc"]}
    assert ids == {"claire-7b-apache", "croissantllm-chat", "mistral-7b-v0.3", "control-low-transfer"}


def test_real_config_lou_candidate_ids_match_tech_spec() -> None:
    config = load_bakeoff_config(REAL_CONFIG_PATH)
    ids = {c.id for c in config["lou"]}
    assert ids == {"mistral-7b-v0.3", "control-low-transfer"}


# --- Config loader: license validation ---------------------------------------


def test_candidate_with_hf_repo_and_no_license_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        "frc:\n  candidates:\n    - id: sketchy\n      hf_repo: \"someorg/some-model\"\n",
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError):
        load_bakeoff_config(bad_config)


def test_control_candidate_with_no_hf_repo_needs_no_license(tmp_path: Path) -> None:
    ok_config = tmp_path / "ok.yml"
    ok_config.write_text(
        "frc:\n  candidates:\n    - id: control-low-transfer\n      base: null\n",
        encoding="utf-8",
    )
    config = load_bakeoff_config(ok_config)
    assert config["frc"][0].id == "control-low-transfer"
    assert config["frc"][0].hf_repo is None


def test_language_key_outside_frc_lou_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(
        "hat:\n  candidates:\n    - id: x\n      base: null\n",
        encoding="utf-8",
    )
    with pytest.raises(BakeoffConfigError):
        load_bakeoff_config(bad_config)


def test_variable_candidate_counts_both_load_without_hardcoded_count_check(tmp_path: Path) -> None:
    # lou with 1 candidate, frc with 5 — the loader must not assume a fixed N.
    config_path = tmp_path / "variable.yml"
    config_path.write_text(
        "frc:\n"
        "  candidates:\n"
        "    - id: a\n      base: null\n"
        "    - id: b\n      base: null\n"
        "    - id: c\n      base: null\n"
        "    - id: d\n      base: null\n"
        "    - id: e\n      base: null\n"
        "lou:\n"
        "  candidates:\n"
        "    - id: f\n      base: null\n",
        encoding="utf-8",
    )
    config = load_bakeoff_config(config_path)
    assert len(config["frc"]) == 5
    assert len(config["lou"]) == 1


# --- run_bakeoff() orchestration ---------------------------------------------


def _candidate(id_: str) -> Candidate:
    return Candidate(id=id_, hf_repo=None, license=None, base=None, note=None)


def test_highest_scoring_non_disqualified_candidate_wins() -> None:
    candidates = [_candidate("low"), _candidate("high")]
    scores = {"low": 0.5, "high": 0.9}

    def fine_tune(c: Candidate) -> object:
        return c.id

    def score(artifact: object, language: str) -> tuple[float, dict[str, float]]:
        return scores[str(artifact)], {}

    def run_red_team(artifact: object, language: str) -> bool:
        return True

    result = run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)
    assert result.winner_candidate_id == "high"


def test_red_team_failure_disqualifies_regardless_of_score() -> None:
    candidates = [_candidate("cheater"), _candidate("honest")]
    scores = {"cheater": 0.99, "honest": 0.5}

    def fine_tune(c: Candidate) -> object:
        return c.id

    def score(artifact: object, language: str) -> tuple[float, dict[str, float]]:
        return scores[str(artifact)], {}

    def run_red_team(artifact: object, language: str) -> bool:
        return str(artifact) != "cheater"

    result = run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)

    cheater_result = next(r for r in result.results if r.candidate_id == "cheater")
    assert cheater_result.disqualified is True
    assert result.winner_candidate_id == "honest"


def test_every_candidate_appears_in_results_not_just_winner() -> None:
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]

    def fine_tune(c: Candidate) -> object:
        return c.id

    def score(artifact: object, language: str) -> tuple[float, dict[str, float]]:
        return 0.5, {}

    def run_red_team(artifact: object, language: str) -> bool:
        return True

    result = run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)
    assert {r.candidate_id for r in result.results} == {"a", "b", "c"}


def test_all_disqualified_yields_no_winner_not_arbitrary_pick() -> None:
    candidates = [_candidate("a"), _candidate("b")]

    def fine_tune(c: Candidate) -> object:
        return c.id

    def score(artifact: object, language: str) -> tuple[float, dict[str, float]]:
        return 0.9, {}

    def run_red_team(artifact: object, language: str) -> bool:
        return False

    result = run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)
    assert result.winner_candidate_id is None
    assert all(r.disqualified for r in result.results)


def test_each_stage_called_exactly_once_per_candidate() -> None:
    candidates = [_candidate("a"), _candidate("b")]
    call_counts = {"fine_tune": 0, "score": 0, "run_red_team": 0}

    def fine_tune(c: Candidate) -> object:
        call_counts["fine_tune"] += 1
        return c.id

    def score(artifact: object, language: str) -> tuple[float, dict[str, float]]:
        call_counts["score"] += 1
        return 0.5, {}

    def run_red_team(artifact: object, language: str) -> bool:
        call_counts["run_red_team"] += 1
        return True

    run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)
    assert call_counts == {"fine_tune": 2, "score": 2, "run_red_team": 2}


def test_run_bakeoff_is_deterministic_given_deterministic_callables() -> None:
    candidates = [_candidate("a"), _candidate("b")]

    def fine_tune(c: Candidate) -> object:
        return c.id

    def score(artifact: object, language: str) -> tuple[float, dict[str, float]]:
        return (0.7 if artifact == "a" else 0.3), {"perplexity": 12.3}

    def run_red_team(artifact: object, language: str) -> bool:
        return True

    first = run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)
    second = run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)
    assert first == second


def test_tied_scores_break_deterministically_by_candidate_id() -> None:
    # No tech-spec section specifies a tie-break rule; the implementation's
    # chosen rule (higher candidate_id string wins) must at least be
    # deterministic and documented, not silently arbitrary per-run.
    candidates = [_candidate("aaa"), _candidate("zzz")]

    def fine_tune(c: Candidate) -> object:
        return c.id

    def score(artifact: object, language: str) -> tuple[float, dict[str, float]]:
        return 0.5, {}  # exact tie

    def run_red_team(artifact: object, language: str) -> bool:
        return True

    result = run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)
    assert result.winner_candidate_id == "zzz"


def test_raw_metrics_passed_through_unchanged() -> None:
    candidates = [_candidate("a")]

    def fine_tune(c: Candidate) -> object:
        return c.id

    def score(artifact: object, language: str) -> tuple[float, dict[str, float]]:
        return 0.5, {"perplexity": 9.1, "grammar_probe_accuracy": 0.62}

    def run_red_team(artifact: object, language: str) -> bool:
        return True

    result = run_bakeoff("frc", {"frc": candidates}, fine_tune=fine_tune, score=score, run_red_team=run_red_team)
    assert result.results[0].raw_metrics == {"perplexity": 9.1, "grammar_probe_accuracy": 0.62}
