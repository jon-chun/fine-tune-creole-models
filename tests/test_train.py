"""Tests for src/train/ — the LoRA/QLoRA/DoRA hyperparameter config loader
(tech-spec v2 §5; MIG-01d, issue #28).

Tests only external behavior: loading fixture/real YAML and asserting on the
returned Hyperparameters/LoadedTrainingConfig or raised error; calling
sweep_learning_rate() and assert_nf4_requires_stretch_arm() directly.
"""

from pathlib import Path

import pytest

from train import (
    TrainingConfigError,
    assert_nf4_requires_stretch_arm,
    load_hyperparameters,
    sweep_learning_rate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG_PATH = REPO_ROOT / "configs" / "training" / "lora_defaults.yml"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(text, encoding="utf-8")
    return path


# _BASE_FIELDS: every v2-required field except rank/epochs/quantization_train
# (each test's own header line varies those three).
_BASE_FIELDS = (
    "method: lora\n"
    "target_modules: all_linear\n"
    "learning_rate: 1.5e-4\n"
    "effective_batch_sequences: 24\n"
    "warmup_fraction: 0.05\n"
    "replay_fraction: 0.15\n"
    "seeds: [42, 43, 44]\n"
    'task_prefixes: ["translate frc->eng"]\n'
)


def _config_text(*, rank: int = 16, epochs: int = 2, quantization: str = "bf16", extra: str = "") -> str:
    return f"rank: {rank}\nepochs: {epochs}\nquantization_train: {quantization}\n{_BASE_FIELDS}{extra}"


# --- Real committed config ---------------------------------------------------


def test_real_config_loads_and_alpha_is_derived_from_rank() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert loaded.hyperparameters.rank == 16
    assert loaded.hyperparameters.alpha == 32


def test_real_config_has_no_warnings() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert loaded.warnings == []


def test_default_config_rank_is_16_alpha_32() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert loaded.hyperparameters.rank == 16
    assert loaded.hyperparameters.alpha == 32


# --- alpha derivation ---------------------------------------------------------


def test_alpha_always_equals_2x_rank(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text(rank=16, epochs=1))
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.alpha == 32


# --- quantization validation: bf16/fp16/8bit/nf4 accepted, else hard gate ------


@pytest.mark.parametrize("bad_value", ["4bit", "4-bit", "garbage"])
def test_unsupported_quantization_is_rejected(tmp_path: Path, bad_value: str) -> None:
    path = _write(tmp_path, _config_text(quantization=bad_value))
    with pytest.raises(TrainingConfigError, match="not supported"):
        load_hyperparameters(path)


@pytest.mark.parametrize("good_value", ["bf16", "fp16", "8bit", "nf4"])
def test_valid_quantization_modes_load_successfully(tmp_path: Path, good_value: str) -> None:
    path = _write(tmp_path, _config_text(quantization=good_value))
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.quantization_train == good_value


# --- nf4 / assert_nf4_requires_stretch_arm (issue #28) -------------------------


def test_nf4_rejected_without_base_size_b() -> None:
    with pytest.raises(TrainingConfigError, match="base_size_b"):
        assert_nf4_requires_stretch_arm("nf4", base_size_b=None)


def test_nf4_rejected_when_base_size_b_below_20() -> None:
    with pytest.raises(TrainingConfigError, match="20"):
        assert_nf4_requires_stretch_arm("nf4", base_size_b=7)


def test_nf4_accepted_when_base_size_b_at_least_20() -> None:
    assert_nf4_requires_stretch_arm("nf4", base_size_b=20)
    assert_nf4_requires_stretch_arm("nf4", base_size_b=32)


def test_non_nf4_quantization_ignores_base_size_b() -> None:
    assert_nf4_requires_stretch_arm("bf16", base_size_b=None)
    assert_nf4_requires_stretch_arm("fp16", base_size_b=1)


# --- epochs: soft warning, 1-3 range (v2 widened) ------------------------------


def test_epochs_1_to_3_loads_without_warning(tmp_path: Path) -> None:
    for epochs in (1, 2, 3):
        path = _write(tmp_path, _config_text(epochs=epochs))
        loaded = load_hyperparameters(path)
        assert not any("epochs" in w for w in loaded.warnings)


def test_epochs_4_loads_with_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text(epochs=4))
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.epochs == 4
    assert any("epochs" in w for w in loaded.warnings)


# --- rank: soft warning, 8-64 range (v2 widened) -------------------------------


def test_rank_8_loads_without_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text(rank=8))
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.rank == 8
    assert not any("rank" in w for w in loaded.warnings)


def test_rank_64_loads_without_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text(rank=64))
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.rank == 64
    assert not any("rank" in w for w in loaded.warnings)


def test_rank_7_loads_with_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text(rank=7))
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.rank == 7
    assert any("rank" in w for w in loaded.warnings)


def test_rank_65_loads_with_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text(rank=65))
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.rank == 65
    assert any("rank" in w for w in loaded.warnings)


# --- method: lora | dora | attention_only (issue #28) --------------------------


@pytest.mark.parametrize("method", ["lora", "dora", "attention_only"])
def test_method_lora_dora_attention_only_load_successfully(tmp_path: Path, method: str) -> None:
    text = (
        f"rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: {method}\n"
        "target_modules: all_linear\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        "seeds: [42, 43, 44]\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.method == method


def test_method_outside_closed_set_raises(tmp_path: Path) -> None:
    text = (
        "rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: full_finetune\n"
        "target_modules: all_linear\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        "seeds: [42, 43, 44]\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="method"):
        load_hyperparameters(path)


def test_method_attention_only_does_not_imply_target_modules(tmp_path: Path) -> None:
    """Judgment call (issue #28): method='attention_only' does not
    auto-derive target_modules='attention_qv_only' — both fields are
    validated independently and a caller sets both explicitly."""
    text = (
        "rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: attention_only\n"
        "target_modules: all_linear\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        "seeds: [42, 43, 44]\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.method == "attention_only"
    assert loaded.hyperparameters.target_modules == "all_linear"


# --- target_modules validation (issue #15) --------------------------------------


def test_invalid_target_modules_raises_training_config_error(tmp_path: Path) -> None:
    text = (
        "rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: lora\n"
        "target_modules: everything\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        "seeds: [42, 43, 44]\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="target_modules"):
        load_hyperparameters(path)


@pytest.mark.parametrize("good_value", ["all_linear", "attention_qv_only"])
def test_valid_target_modules_load_successfully(tmp_path: Path, good_value: str) -> None:
    text = (
        f"rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: lora\n"
        f"target_modules: {good_value}\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        "seeds: [42, 43, 44]\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.target_modules == good_value


def test_missing_target_modules_raises_training_config_error(tmp_path: Path) -> None:
    text = (
        "rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: lora\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        "seeds: [42, 43, 44]\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="target_modules"):
        load_hyperparameters(path)


# --- effective_batch_sequences: soft warning, 16-32 range (replaces batch_size_strategy) --


def test_effective_batch_sequences_16_to_32_loads_without_warning(tmp_path: Path) -> None:
    for value in (16, 24, 32):
        text = _config_text().replace(
            "effective_batch_sequences: 24", f"effective_batch_sequences: {value}"
        )
        path = _write(tmp_path, text)
        loaded = load_hyperparameters(path)
        assert not any("effective_batch_sequences" in w for w in loaded.warnings)


def test_effective_batch_sequences_outside_range_loads_with_warning(tmp_path: Path) -> None:
    text = _config_text().replace("effective_batch_sequences: 24", "effective_batch_sequences: 8")
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.effective_batch_sequences == 8
    assert any("effective_batch_sequences" in w for w in loaded.warnings)


def test_missing_effective_batch_sequences_raises_training_config_error(tmp_path: Path) -> None:
    text = (
        "rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: lora\n"
        "target_modules: all_linear\n"
        "learning_rate: 1.5e-4\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        "seeds: [42, 43, 44]\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="effective_batch_sequences"):
        load_hyperparameters(path)


# --- replay_fraction: soft warning, 10-20% band (issue #28) ---------------------


def test_replay_fraction_within_10_to_20_percent_loads_without_warning(tmp_path: Path) -> None:
    for value in (0.10, 0.15, 0.20):
        text = _config_text().replace("replay_fraction: 0.15", f"replay_fraction: {value}")
        path = _write(tmp_path, text)
        loaded = load_hyperparameters(path)
        assert not any("replay_fraction" in w for w in loaded.warnings)


def test_replay_fraction_outside_band_loads_with_warning(tmp_path: Path) -> None:
    text = _config_text().replace("replay_fraction: 0.15", "replay_fraction: 0.30")
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.replay_fraction == pytest.approx(0.30)
    assert any("replay_fraction" in w for w in loaded.warnings)


def test_missing_replay_fraction_raises_training_config_error(tmp_path: Path) -> None:
    text = (
        "rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: lora\n"
        "target_modules: all_linear\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "seeds: [42, 43, 44]\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="replay_fraction"):
        load_hyperparameters(path)


# --- seeds: non-empty list required (issue #28) ---------------------------------


def test_seeds_list_of_three_loads_successfully(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text())
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.seeds == (42, 43, 44)


def test_seeds_empty_list_raises(tmp_path: Path) -> None:
    text = _config_text().replace("seeds: [42, 43, 44]", "seeds: []")
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="seeds"):
        load_hyperparameters(path)


def test_missing_seeds_raises_training_config_error(tmp_path: Path) -> None:
    text = (
        "rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: lora\n"
        "target_modules: all_linear\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        'task_prefixes: ["translate frc->eng"]\n'
    )
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="seeds"):
        load_hyperparameters(path)


# --- early_stopping: optional, patience >= 1 (issue #28) ------------------------


def test_early_stopping_patience_at_least_one(tmp_path: Path) -> None:
    text = _config_text(extra="early_stopping:\n  metric: dev_chrf_plus_plus\n  patience: 1\n")
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.early_stopping is not None
    assert loaded.hyperparameters.early_stopping.metric == "dev_chrf_plus_plus"
    assert loaded.hyperparameters.early_stopping.patience == 1


def test_early_stopping_patience_below_one_raises(tmp_path: Path) -> None:
    text = _config_text(extra="early_stopping:\n  metric: dev_chrf_plus_plus\n  patience: 0\n")
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="patience"):
        load_hyperparameters(path)


def test_early_stopping_absent_loads_as_none(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text())
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.early_stopping is None


# --- task_prefixes (issue #28) ---------------------------------------------------


def test_task_prefixes_default_matches_tech_spec_regime() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert loaded.hyperparameters.task_prefixes == (
        "translate frc->eng",
        "translate frc->fra",
        "normalize",
        "contrast",
        "judge",
    )


def test_task_prefixes_empty_raises(tmp_path: Path) -> None:
    text = _config_text().replace('task_prefixes: ["translate frc->eng"]', "task_prefixes: []")
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="task_prefixes"):
        load_hyperparameters(path)


def test_missing_task_prefixes_raises(tmp_path: Path) -> None:
    text = (
        "rank: 16\nepochs: 2\nquantization_train: bf16\nmethod: lora\n"
        "target_modules: all_linear\n"
        "learning_rate: 1.5e-4\n"
        "effective_batch_sequences: 24\n"
        "warmup_fraction: 0.05\n"
        "replay_fraction: 0.15\n"
        "seeds: [42, 43, 44]\n"
    )
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="task_prefixes"):
        load_hyperparameters(path)


# --- learning_rate: soft warning, 1e-4 to 2e-4 band (issue #28) -----------------


def test_learning_rate_within_1e4_to_2e4_loads_without_warning(tmp_path: Path) -> None:
    for value in ("1.0e-4", "1.5e-4", "2.0e-4"):
        text = _config_text().replace("learning_rate: 1.5e-4", f"learning_rate: {value}")
        path = _write(tmp_path, text)
        loaded = load_hyperparameters(path)
        assert not any("learning_rate" in w for w in loaded.warnings)


def test_learning_rate_outside_band_loads_with_warning(tmp_path: Path) -> None:
    text = _config_text().replace("learning_rate: 1.5e-4", "learning_rate: 5.0e-4")
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.learning_rate == pytest.approx(5.0e-4)
    assert any("learning_rate" in w for w in loaded.warnings)


# --- max_seq_len: soft warning, 1024-2048 band (issue #28), on top of the -------
# --- existing hard corpus-derived-knob presence requirement --------------------


def test_max_seq_len_outside_1024_to_2048_loads_with_warning(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}"
        "max_seq_len: 512\n"
        "batch_token_budget: 8192\n"
        "max_train_tokens: 2000000\n"
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert any("max_seq_len" in w for w in loaded.warnings)


def test_max_seq_len_within_1024_to_2048_loads_without_warning(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}"
        "max_seq_len: 1536\n"
        "batch_token_budget: 8192\n"
        "max_train_tokens: 2000000\n"
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert not any("max_seq_len" in w for w in loaded.warnings)


# --- sweep_learning_rate() -----------------------------------------------------


def test_sweep_spans_approximately_one_order_of_magnitude() -> None:
    values = sweep_learning_rate(1e-4)
    assert min(values) == pytest.approx(1e-5, rel=0.05)
    assert max(values) == pytest.approx(1e-3, rel=0.05)


def test_sweep_includes_base_rate() -> None:
    values = sweep_learning_rate(1e-4)
    assert any(v == pytest.approx(1e-4) for v in values)


def test_sweep_is_deterministic() -> None:
    first = sweep_learning_rate(1e-4)
    second = sweep_learning_rate(1e-4)
    assert first == second


# --- determinism of loading -----------------------------------------------------


def test_loading_same_file_twice_is_equal() -> None:
    first = load_hyperparameters(REAL_CONFIG_PATH)
    second = load_hyperparameters(REAL_CONFIG_PATH)
    assert first == second


# --- seeds on the real config (tech-spec §10 reproducibility contract) ---------


def test_real_config_has_seeds() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert isinstance(loaded.hyperparameters.seeds, tuple)
    assert len(loaded.hyperparameters.seeds) >= 1
    assert all(isinstance(s, int) for s in loaded.hyperparameters.seeds)


# --- lr_sweep_steps: YAML-configurable sweep_learning_rate() step count --------


def test_real_config_lr_sweep_steps_matches_sweep_default() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert loaded.lr_sweep_steps == 5


def test_missing_lr_sweep_steps_falls_back_to_default(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text())
    loaded = load_hyperparameters(path)
    assert loaded.lr_sweep_steps == 5


# --- per-language overrides -----------------------------------------------------


_CORPUS_KNOBS = "max_seq_len: 1536\nbatch_token_budget: 8192\nmax_train_tokens: 2000000\n"


def test_language_override_is_applied(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}{_CORPUS_KNOBS}"
        "overrides:\n"
        "  frc:\n"
        "    rank: 32\n"
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path, language="frc")
    assert loaded.hyperparameters.rank == 32
    assert loaded.hyperparameters.alpha == 64


def test_language_override_absent_for_other_language(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}{_CORPUS_KNOBS}"
        "overrides:\n"
        "  frc:\n"
        "    rank: 32\n"
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path, language="lou")
    assert loaded.hyperparameters.rank == 16


def test_no_language_argument_ignores_overrides(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}"
        "overrides:\n"
        "  frc:\n"
        "    rank: 32\n"
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.rank == 16


# --- missing keys: rank/epochs/learning_rate/warmup_fraction (issue #15) --------


def test_missing_rank_raises_training_config_error(tmp_path: Path) -> None:
    text = f"epochs: 2\nquantization_train: bf16\n{_BASE_FIELDS}"
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError, match="rank"):
        load_hyperparameters(path)


# --- corpus-derived knobs: max_seq_len / batch_token_budget / max_train_tokens (issue #20) ---


def test_corpus_knobs_absent_and_no_language_load_as_none(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_text())
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.max_seq_len is None
    assert loaded.hyperparameters.batch_token_budget is None
    assert loaded.hyperparameters.max_train_tokens is None


def test_corpus_derived_knob_requirement_unchanged(tmp_path: Path) -> None:
    """Regression (issue #28 Out of scope item 11): rerun the pre-MIG-01d
    language_requested_with_no_overrides_raises_naming_all_three fixture
    verbatim to confirm this migration made no behavior change here."""
    path = _write(tmp_path, _config_text())
    with pytest.raises(TrainingConfigError, match="frc") as exc_info:
        load_hyperparameters(path, language="frc")
    message = str(exc_info.value)
    assert "max_seq_len" in message
    assert "batch_token_budget" in message
    assert "max_train_tokens" in message


def test_language_requested_with_partial_overrides_raises_naming_missing_only(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}"
        "overrides:\n"
        "  frc:\n"
        "    max_seq_len: 1536\n"
    )
    path = _write(tmp_path, text)
    with pytest.raises(TrainingConfigError) as exc_info:
        load_hyperparameters(path, language="frc")
    message = str(exc_info.value)
    assert "batch_token_budget" in message
    assert "max_train_tokens" in message


def test_language_requested_with_all_three_overrides_loads_successfully(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}"
        "overrides:\n"
        "  frc:\n"
        "    max_seq_len: 1536\n"
        "    batch_token_budget: 8192\n"
        "    max_train_tokens: 2000000\n"
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path, language="frc")
    assert loaded.hyperparameters.max_seq_len == 1536
    assert loaded.hyperparameters.batch_token_budget == 8192
    assert loaded.hyperparameters.max_train_tokens == 2000000


@pytest.mark.parametrize("knob", ["max_seq_len", "batch_token_budget", "max_train_tokens"])
def test_corpus_knob_non_positive_raises(tmp_path: Path, knob: str) -> None:
    path = _write(tmp_path, f"{_config_text()}{knob}: 0\n")
    with pytest.raises(TrainingConfigError, match=knob):
        load_hyperparameters(path)


def test_max_train_tokens_less_than_batch_token_budget_warns(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}"
        "max_seq_len: 1536\n"
        "batch_token_budget: 8192\n"
        "max_train_tokens: 100\n"
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert any("max_train_tokens" in w and "batch_token_budget" in w for w in loaded.warnings)


def test_max_train_tokens_at_least_batch_token_budget_does_not_warn(tmp_path: Path) -> None:
    text = (
        f"{_config_text()}"
        "max_seq_len: 1536\n"
        "batch_token_budget: 8192\n"
        "max_train_tokens: 2000000\n"
    )
    path = _write(tmp_path, text)
    loaded = load_hyperparameters(path)
    assert not any("batch_token_budget" in w for w in loaded.warnings)


def test_real_config_corpus_knobs_are_none_without_language() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert loaded.hyperparameters.max_seq_len is None
    assert loaded.hyperparameters.batch_token_budget is None
    assert loaded.hyperparameters.max_train_tokens is None
