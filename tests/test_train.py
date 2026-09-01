"""Tests for src/train/ — the LoRA/QLoRA/DoRA hyperparameter config loader
(tech-spec §5).

Tests only external behavior: loading fixture/real YAML and asserting on the
returned Hyperparameters/LoadedTrainingConfig or raised error; calling
sweep_learning_rate() and asserting on the returned list.
"""

from pathlib import Path

import pytest

from train import (
    TrainingConfigError,
    load_hyperparameters,
    sweep_learning_rate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG_PATH = REPO_ROOT / "configs" / "training" / "lora_defaults.yml"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(text, encoding="utf-8")
    return path


_BASE_FIELDS = (
    "target_modules: all_linear\n"
    "learning_rate: 1.0e-4\n"
    "batch_size_strategy: largest_that_fits\n"
    "warmup_fraction: 0.05\n"
)


# --- Real committed config ---------------------------------------------------


def test_real_config_loads_and_alpha_is_derived_from_rank() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert loaded.hyperparameters.rank == 24
    assert loaded.hyperparameters.alpha == 48


def test_real_config_has_no_warnings() -> None:
    loaded = load_hyperparameters(REAL_CONFIG_PATH)
    assert loaded.warnings == []


# --- alpha derivation ---------------------------------------------------------


def test_alpha_always_equals_2x_rank(tmp_path: Path) -> None:
    path = _write(tmp_path, f"rank: 16\nepochs: 1\nquantization_train: bf16\n{_BASE_FIELDS}")
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.alpha == 32


# --- quantization validation: hard gate ---------------------------------------


@pytest.mark.parametrize("bad_value", ["4bit", "4-bit"])
def test_4bit_quantization_is_rejected(tmp_path: Path, bad_value: str) -> None:
    path = _write(tmp_path, f"rank: 24\nepochs: 2\nquantization_train: {bad_value}\n{_BASE_FIELDS}")
    with pytest.raises(TrainingConfigError, match="adapter merging"):
        load_hyperparameters(path)


@pytest.mark.parametrize("good_value", ["bf16", "fp16", "8bit"])
def test_valid_quantization_modes_load_successfully(tmp_path: Path, good_value: str) -> None:
    path = _write(tmp_path, f"rank: 24\nepochs: 2\nquantization_train: {good_value}\n{_BASE_FIELDS}")
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.quantization_train == good_value


# --- epochs: soft warning, not a hard gate -------------------------------------


def test_epochs_outside_1_to_2_loads_with_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, f"rank: 24\nepochs: 5\nquantization_train: bf16\n{_BASE_FIELDS}")
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.epochs == 5
    assert any("epochs" in w for w in loaded.warnings)


def test_epochs_within_1_to_2_loads_without_epoch_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, f"rank: 24\nepochs: 2\nquantization_train: bf16\n{_BASE_FIELDS}")
    loaded = load_hyperparameters(path)
    assert not any("epochs" in w for w in loaded.warnings)


# --- rank: soft warning, not a hard gate ---------------------------------------


def test_rank_outside_16_to_32_loads_with_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, f"rank: 64\nepochs: 2\nquantization_train: bf16\n{_BASE_FIELDS}")
    loaded = load_hyperparameters(path)
    assert loaded.hyperparameters.rank == 64
    assert any("rank" in w for w in loaded.warnings)


def test_rank_within_16_to_32_loads_without_rank_warning(tmp_path: Path) -> None:
    path = _write(tmp_path, f"rank: 24\nepochs: 2\nquantization_train: bf16\n{_BASE_FIELDS}")
    loaded = load_hyperparameters(path)
    assert not any("rank" in w for w in loaded.warnings)


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
