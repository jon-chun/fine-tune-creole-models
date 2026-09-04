"""Tests for src/eval/metrics.py — tech-spec v2 §6.1 Layer-B metric
implementations (issue #34): chrF/chrF++, corpus BLEU, WER/CER decomposed
against the two reference layers, token error rate, tokenizer fertility,
per-cell accuracy with Wilson CIs, the paired-item bootstrap CI, the Q13
`within_noise` rule (ADR 0015), and the `compute_metrics`/`headline_metric`
seam issue #23 row D2 asked for.

Fixture values are hand-computed and pinned in comments next to each
assertion, per the wave-2 preamble's "hand-computed fixture tests" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.metrics import (
    CellAccuracy,
    MetricReport,
    char_error_rate,
    chrf,
    chrf_plus_plus,
    compute_metrics,
    corpus_bleu,
    headline_metric,
    paired_bootstrap_ci,
    per_cell_accuracy_with_wilson_ci,
    token_error_rate,
    tokenizer_fertility,
    wer_cer_decomposed,
    wilson_confidence_interval,
    within_noise,
    word_error_rate,
)


# --- chrF / chrF++ --------------------------------------------------------------


def test_chrf_plus_plus_perfect_match_is_100() -> None:
    assert chrf_plus_plus("hello world", "hello world") == pytest.approx(100.0)
    assert chrf("hello world", "hello world") == pytest.approx(100.0)


def test_chrf_plus_plus_hand_computed_fixture() -> None:
    """hyp="ab", ref="ac", max_char_ngram=2, max_word_ngram=1.
    Char n=1: hyp={a,b}, ref={a,c}; overlap={a:1}; P=1/2, R=1/2;
      F2 = (1+4)*P*R / (4P+R) = 5*0.25/2.5 = 0.5.
    Char n=2: hyp={"ab"}, ref={"ac"}; overlap=0 -> F2=0.0.
    Word n=1: hyp=["ab"], ref=["ac"]; overlap=0 -> F2=0.0.
    Mean of (0.5, 0.0, 0.0) = 0.16666... ; *100 = 16.6666...
    """
    score = chrf_plus_plus("ab", "ac", max_char_ngram=2, max_word_ngram=1)
    assert score == pytest.approx(16.666666666666664)

    # chrf() alone (char n-grams only, same two orders): mean(0.5, 0.0)*100 = 25.0
    assert chrf("ab", "ac", max_char_ngram=2) == pytest.approx(25.0)


# --- corpus BLEU -----------------------------------------------------------------


def test_bleu_smoothing_nonzero_on_short_hypothesis() -> None:
    """A 1-word hypothesis against a 6-word reference has zero 2/3/4-gram
    overlap (and even zero 1-gram overlap here), which would drive
    unsmoothed BLEU to exactly 0.0. Additive smoothing keeps the score
    strictly positive."""
    score = corpus_bleu(["the"], ["the cat sat on the mat"])
    assert score > 0.0
    assert score == pytest.approx(0.6737946999085467)


def test_bleu_perfect_match_is_100() -> None:
    score = corpus_bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert score == pytest.approx(100.0)


def test_bleu_requires_equal_length_inputs() -> None:
    with pytest.raises(ValueError):
        corpus_bleu(["a", "b"], ["a"])


# --- WER / CER / TER --------------------------------------------------------------


def test_wer_cer_decomposed_diplomatic_vs_normalized() -> None:
    """hyp="le chat noir" vs diplomatic="le chat noir diplomatic-ish" (5
    words) and normalized="le chat noir" (exact match).
    WER diplomatic: edit distance(["le","chat","noir"],
      ["le","chat","noir","diplomatic-ish"]) = 1 deletion / 4 ref words = 0.25.
    WER normalized: identical -> 0.0.
    CER diplomatic: edit distance over characters / len(reference) =
      12 / 27 = 0.5555...; CER normalized: identical -> 0.0.
    """
    result = wer_cer_decomposed(
        "le chat noir", "le chat noir diplomatic-ish", "le chat noir"
    )
    assert result.wer_diplomatic == pytest.approx(0.25)
    assert result.wer_normalized == pytest.approx(0.0)
    assert result.cer_diplomatic == pytest.approx(0.5555555555555556)
    assert result.cer_normalized == pytest.approx(0.0)


def test_word_error_rate_hand_fixture() -> None:
    # hyp/ref differ by one substitution ("chat"->"chien") over 2 ref words.
    assert word_error_rate("le chat", "le chien") == pytest.approx(0.5)


def test_char_error_rate_identical_strings_is_zero() -> None:
    assert char_error_rate("abc", "abc") == pytest.approx(0.0)


def test_token_error_rate_for_normalize_task() -> None:
    """token_error_rate is word_error_rate under a task-specific name
    (tech-spec v2 §6.1's normalize-task headline): one substitution over 2
    reference words = 0.5."""
    assert token_error_rate("le chat", "le chien") == pytest.approx(0.5)


def test_word_error_rate_empty_reference_nonempty_hypothesis_is_capped_at_one() -> None:
    assert word_error_rate("extra words", "") == 1.0


def test_word_error_rate_both_empty_is_zero() -> None:
    assert word_error_rate("", "") == 0.0


# --- tokenizer fertility -----------------------------------------------------------


def test_tokenizer_fertility_tokens_per_word() -> None:
    """tokenizer(t) here returns one token per character (spaces removed).
    "hello world" -> 10 char-tokens over 2 words; "foo" -> 3 char-tokens
    over 1 word. total tokens = 13, total words = 3 -> 13/3 = 4.3333...
    """
    fertility = tokenizer_fertility(
        ["hello world", "foo"], lambda text: list(text.replace(" ", ""))
    )
    assert fertility == pytest.approx(13 / 3)


def test_tokenizer_fertility_raises_on_all_empty_texts() -> None:
    with pytest.raises(ValueError):
        tokenizer_fertility(["", "   "], lambda text: list(text))


# --- Wilson CI / per-cell accuracy --------------------------------------------------


def test_wilson_ci_hand_fixture() -> None:
    """successes=8, n=10, z=1.959963984540054 (95% two-sided).
    p_hat=0.8; verified against a reference Wilson-interval calculation."""
    lower, upper = wilson_confidence_interval(8, 10)
    assert lower == pytest.approx(0.4901624715366418)
    assert upper == pytest.approx(0.9433178485456247)


def test_wilson_ci_rejects_n_zero() -> None:
    with pytest.raises(ValueError):
        wilson_confidence_interval(0, 0)


def test_per_cell_accuracy_with_wilson_ci() -> None:
    """cell A: 1/2 correct -> accuracy 0.5; cell B: 1/1 correct -> accuracy 1.0.
    CIs are the same Wilson computation as test_wilson_ci_hand_fixture's
    style, pinned against the same implementation."""
    result = per_cell_accuracy_with_wilson_ci(["A", "A", "B"], [True, False, True])
    assert set(result) == {"A", "B"}
    assert isinstance(result["A"], CellAccuracy)
    assert result["A"].n == 2
    assert result["A"].accuracy == pytest.approx(0.5)
    assert result["A"].ci_lower == pytest.approx(0.09453120573423071)
    assert result["A"].ci_upper == pytest.approx(0.9054687942657693)
    assert result["B"].n == 1
    assert result["B"].accuracy == pytest.approx(1.0)


def test_per_cell_accuracy_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        per_cell_accuracy_with_wilson_ci(["A"], [True, False])


# --- paired bootstrap CI and within_noise (ADR 0015 Q13) ----------------------------


def test_paired_bootstrap_ci_excludes_zero_on_clear_gain() -> None:
    """a is uniformly 20 points above b on every item -> every resampled
    mean delta is exactly 20.0, so the CI is the degenerate point (20, 20),
    which excludes zero by a wide margin."""
    scores_a = [90.0] * 20
    scores_b = [70.0] * 20
    lower, upper = paired_bootstrap_ci(scores_a, scores_b, seed=42, n_resamples=1000)
    assert lower > 0.0
    assert upper > 0.0
    assert lower == pytest.approx(20.0)
    assert upper == pytest.approx(20.0)


def test_paired_bootstrap_ci_requires_equal_length() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [1.0], seed=1)


def test_paired_bootstrap_ci_requires_nonempty() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [], seed=1)


def test_within_noise_q13_rule_true_when_ci_contains_zero() -> None:
    """Two series with matched means and symmetric per-item noise around
    the same center (a - b oscillates around 0 by design) -> the paired
    bootstrap CI of the mean delta contains zero -> within_noise is True
    (ADR 0015 Q13: interval containing zero means "within noise")."""
    scores_a = [50.0, 52.0, 48.0, 51.0, 49.0, 50.5, 49.5, 50.0, 51.5, 48.5]
    scores_b = [50.0, 48.0, 52.0, 49.0, 51.0, 49.5, 50.5, 50.0, 48.5, 51.5]
    assert within_noise(scores_a, scores_b, seed=7) is True


def test_within_noise_q13_rule_false_on_clear_gain() -> None:
    scores_a = [90.0] * 20
    scores_b = [70.0] * 20
    assert within_noise(scores_a, scores_b, seed=42) is False


def test_within_noise_is_deterministic_given_same_seed() -> None:
    scores_a = [50.0, 52.0, 48.0, 51.0, 49.0]
    scores_b = [50.0, 48.0, 52.0, 49.0, 51.0]
    first = within_noise(scores_a, scores_b, seed=3)
    second = within_noise(scores_a, scores_b, seed=3)
    assert first == second


# --- headline_metric / compute_metrics (issue #23 row D2) --------------------------


@dataclass
class _FakeEvalItem:
    """A minimal stand-in satisfying eval.metrics's structural
    `_EvalItemLike` shape, avoiding any dependency on src/eval/__init__.py's
    real `EvalItem` from this metrics-only test file (that cross-check is
    test_eval_item_gains_source_text_and_references in tests/test_eval.py,
    which imports the real EvalItem)."""

    item_id: str
    reference: str | None = None
    reference_diplomatic: str | None = None
    cell_id: str | None = None


def test_headline_metric_per_task() -> None:
    assert headline_metric("mt") == "chrf_plus_plus"
    assert headline_metric("normalize") == "token_error_rate"
    assert headline_metric("contrast") == "per_cell_accuracy"
    assert headline_metric("judge") == "per_cell_accuracy"


def test_headline_metric_rejects_unknown_task() -> None:
    with pytest.raises(ValueError):
        headline_metric("not_a_real_task")  # type: ignore[arg-type]


def test_compute_metrics_mt_task_headline_is_chrf_plus_plus() -> None:
    items = [
        _FakeEvalItem("a", reference="bonjour le monde"),
        _FakeEvalItem("b", reference="comment ça va"),
    ]
    hypotheses = ["bonjour le monde", "comment ca va"]
    report = compute_metrics(items, hypotheses, "mt")
    assert isinstance(report, MetricReport)
    assert report.task == "mt"
    assert report.headline_name == "chrf_plus_plus"
    assert report.headline_value == pytest.approx(report.metrics["chrf_plus_plus"])
    assert "bleu" in report.metrics
    assert "chrf" in report.metrics


def test_compute_metrics_normalize_task_includes_wer_cer_when_diplomatic_present() -> None:
    items = [
        _FakeEvalItem("a", reference="bonjour", reference_diplomatic="Bonjour!"),
        _FakeEvalItem("b", reference="ca va", reference_diplomatic="Ça va?"),
    ]
    hypotheses = ["bonjour", "ca va"]
    report = compute_metrics(items, hypotheses, "normalize")
    assert report.headline_name == "token_error_rate"
    assert report.headline_value == pytest.approx(0.0)
    assert "wer_diplomatic" in report.metrics
    assert "cer_diplomatic" in report.metrics


def test_compute_metrics_normalize_task_omits_wer_cer_without_diplomatic_layer() -> None:
    items = [_FakeEvalItem("a", reference="bonjour")]
    report = compute_metrics(items, ["bonjour"], "normalize")
    assert "wer_diplomatic" not in report.metrics
    assert report.metrics["token_error_rate"] == pytest.approx(0.0)


def test_compute_metrics_contrast_task_headline_is_per_cell_accuracy() -> None:
    items = [
        _FakeEvalItem("a", reference="yes", cell_id="VERB-001"),
        _FakeEvalItem("b", reference="no", cell_id="VERB-001"),
        _FakeEvalItem("c", reference="yes", cell_id="PRO-002"),
    ]
    hypotheses = ["yes", "yes", "yes"]
    report = compute_metrics(items, hypotheses, "contrast")
    assert report.headline_name == "per_cell_accuracy"
    assert report.headline_value == pytest.approx(2 / 3)
    assert set(report.per_cell) == {"VERB-001", "PRO-002"}
    assert report.per_cell["VERB-001"].accuracy == pytest.approx(0.5)
    assert report.per_cell["PRO-002"].accuracy == pytest.approx(1.0)


def test_compute_metrics_judge_task_uses_same_cell_accuracy_path_as_contrast() -> None:
    items = [_FakeEvalItem("a", reference="accept", cell_id="ANTI-HAT-001")]
    report = compute_metrics(items, ["accept"], "judge")
    assert report.task == "judge"
    assert report.headline_name == "per_cell_accuracy"
    assert report.headline_value == pytest.approx(1.0)


def test_compute_metrics_requires_equal_length_items_and_hypotheses() -> None:
    items = [_FakeEvalItem("a", reference="x")]
    with pytest.raises(ValueError):
        compute_metrics(items, ["x", "y"], "mt")


def test_compute_metrics_requires_nonempty_items() -> None:
    with pytest.raises(ValueError):
        compute_metrics([], [], "mt")


def test_compute_metrics_mt_task_requires_reference() -> None:
    items = [_FakeEvalItem("a", reference=None)]
    with pytest.raises(ValueError, match="a"):
        compute_metrics(items, ["some text"], "mt")


def test_compute_metrics_contrast_task_requires_cell_id() -> None:
    items = [_FakeEvalItem("a", reference="yes", cell_id=None)]
    with pytest.raises(ValueError, match="a"):
        compute_metrics(items, ["yes"], "contrast")


def test_compute_metrics_rejects_unsupported_task() -> None:
    items = [_FakeEvalItem("a", reference="x")]
    with pytest.raises(ValueError):
        compute_metrics(items, ["x"], "not_a_real_task")  # type: ignore[arg-type]
