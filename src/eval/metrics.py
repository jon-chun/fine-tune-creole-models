"""Layer-B metric implementations (tech-spec v2 §6.1) — stdlib only.

This module answers issue #23 row D2's question ("EvalItem has no
text/reference; which metric is the headline float") for the metrics
tech-spec v2 §6.1 lists as computable without external models or a gold set:
chrF / chrF++, corpus BLEU with smoothing, token error rate (normalization
task), WER/CER decomposed against the diplomatic and normalized reference
layers, tokenizer fertility, per-cell accuracy with Wilson 95% CIs, and a
paired item-resampling bootstrap CI for deltas between two metric series —
including the Q13 `within_noise` rule (ADR 0015).

Out of scope (tech-spec v2 §6.1/§6.2, this ticket's "Out of scope"):
COMET/xCOMET (needs a trained model), human dialect-authenticity ratings
(needs a native reviewer), LID macro-F1/confusion-pair F1 (`src/lid`'s own
job), speech metrics (`src/speech_eval`, tech-spec v2 §9), a real gold set,
and the `score` seam adapter onto `src/bakeoff.ScoreResult` (Wave 3 wiring,
issue #23 row D2).

Every public function is pure: no I/O, no hidden randomness except where a
`seed` parameter is given (bootstrap resampling), so results are
reproducible and independently unit-testable per CLAUDE.md's wave-dispatch
discipline.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

from data_contract import LanguageTag

_NgramT = TypeVar("_NgramT", bound=Hashable)

# The five Layer-B task families this module scores (tech-spec v2 §6.1's
# per-task metrics table; the probe-schema `task` enum reused rather than
# redefined — tech-spec v2 §6.3 "Probe schema" — with `normalize` and
# `translate_eng`/`translate_fra` collapsed to `mt`/`normalize` here since
# this module's job is "which metric family", not which target language).
Task = Literal["mt", "normalize", "contrast", "judge"]

__all__ = [
    "Task",
    "MetricReport",
    "chrf",
    "chrf_plus_plus",
    "corpus_bleu",
    "word_error_rate",
    "char_error_rate",
    "wer_cer_decomposed",
    "token_error_rate",
    "tokenizer_fertility",
    "per_cell_accuracy_with_wilson_ci",
    "wilson_confidence_interval",
    "paired_bootstrap_ci",
    "within_noise",
    "compute_metrics",
    "headline_metric",
]

_CHRF_BETA = 2.0
_CHRF_CHAR_NGRAM_MAX = 6
_CHRF_WORD_NGRAM_MAX = 2
_DEFAULT_BOOTSTRAP_RESAMPLES = 1000
_DEFAULT_WILSON_Z = 1.959963984540054  # z for 95% two-sided (tech-spec v2 §6.1/§6.3)


# --- shared n-gram helpers -----------------------------------------------------


def _char_ngrams(text: str, n: int) -> Counter[str]:
    """Character n-grams over `text` with no whitespace stripped (chrF is a
    character-level metric operated on the raw string, tech-spec v2 §6.1)."""
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _word_ngrams(words: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if len(words) < n:
        return Counter()
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def _ngram_f_score(
    hyp_ngrams: Counter[_NgramT], ref_ngrams: Counter[_NgramT], *, beta: float
) -> tuple[float, float, float]:
    """Precision, recall, and the F-beta combination for one n-gram order,
    given hypothesis and reference n-gram multisets. Returns (precision,
    recall, f_beta), each 0.0 when its denominator is empty (no n-grams of
    this order exist in that side) rather than raising, since chrF averages
    across many orders and a single empty order must not crash the whole
    computation."""
    if not hyp_ngrams and not ref_ngrams:
        return 1.0, 1.0, 1.0
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    hyp_total = sum(hyp_ngrams.values())
    ref_total = sum(ref_ngrams.values())
    precision = overlap / hyp_total if hyp_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    if precision == 0.0 and recall == 0.0:
        return precision, recall, 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    f_beta = (1 + beta_sq) * precision * recall / denom if denom else 0.0
    return precision, recall, f_beta


# --- chrF / chrF++ (tech-spec v2 §6.1: "MT ... chrF++ primary") ----------------


def chrf(hypothesis: str, reference: str, *, max_char_ngram: int = _CHRF_CHAR_NGRAM_MAX) -> float:
    """chrF (Popović 2015): the mean F-beta (beta=2, tech-spec v2 §6.1's
    "chrF++") over character n-gram orders 1..max_char_ngram, on a 0-100
    scale. Character-only — no word n-grams (see chrf_plus_plus for the
    "++" word-n-gram extension). A perfect match scores exactly 100.0.
    """
    scores = []
    for n in range(1, max_char_ngram + 1):
        _, _, f_beta = _ngram_f_score(_char_ngrams(hypothesis, n), _char_ngrams(reference, n), beta=_CHRF_BETA)
        scores.append(f_beta)
    return 100.0 * (sum(scores) / len(scores) if scores else 0.0)


def chrf_plus_plus(
    hypothesis: str,
    reference: str,
    *,
    max_char_ngram: int = _CHRF_CHAR_NGRAM_MAX,
    max_word_ngram: int = _CHRF_WORD_NGRAM_MAX,
) -> float:
    """chrF++ (Popović 2017, tech-spec v2 §6.1's MT headline metric): chrF's
    character n-grams (orders 1..max_char_ngram) plus word n-grams (orders
    1..max_word_ngram) averaged into a single F-beta (beta=2) score on a
    0-100 scale. Word n-grams are computed over whitespace-split tokens; a
    perfect match (identical hypothesis and reference) scores exactly 100.0
    regardless of content, since every order's precision and recall are 1.0.
    """
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    scores = []
    for n in range(1, max_char_ngram + 1):
        _, _, f_beta = _ngram_f_score(_char_ngrams(hypothesis, n), _char_ngrams(reference, n), beta=_CHRF_BETA)
        scores.append(f_beta)
    for n in range(1, max_word_ngram + 1):
        _, _, f_beta = _ngram_f_score(_word_ngrams(hyp_words, n), _word_ngrams(ref_words, n), beta=_CHRF_BETA)
        scores.append(f_beta)
    return 100.0 * (sum(scores) / len(scores) if scores else 0.0)


# --- corpus BLEU with smoothing (tech-spec v2 §6.1: "BLEU secondary") ----------


def _brevity_penalty(hyp_len: int, ref_len: int) -> float:
    if hyp_len == 0:
        return 0.0
    if hyp_len >= ref_len:
        return 1.0
    return math.exp(1 - ref_len / hyp_len)


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str], *, max_ngram: int = 4) -> float:
    """Corpus-level BLEU (Papineni et al. 2002) on a 0-100 scale, with
    additive ("+1") smoothing on each n-gram order's precision (Lin & Och
    2004 add-one smoothing, chosen — over leaving zero-count orders
    undefined — specifically so a short hypothesis with no 4-gram overlap
    still yields a nonzero score rather than an undefined/zero BLEU,
    matching this ticket's acceptance test name). `hypotheses` and
    `references` must be the same length, one hypothesis-reference pair per
    corpus item; word-tokenized by whitespace split.

    Corpus-level, not sentence-averaged (tech-spec v2 §6.1 names "corpus
    BLEU" implicitly by citing it alongside chrF++ as an MT metric reported
    over a whole test split, and corpus BLEU is the standard MT
    leaderboard convention this project's readers expect)."""
    if len(hypotheses) != len(references):
        raise ValueError("corpus_bleu requires equal-length hypotheses and references")
    if not hypotheses:
        return 0.0

    hyp_len_total = 0
    ref_len_total = 0
    precisions = []
    for n in range(1, max_ngram + 1):
        clipped_overlap = 0
        hyp_total = 0
        for hyp, ref in zip(hypotheses, references, strict=True):
            hyp_words = hyp.split()
            ref_words = ref.split()
            hyp_ngrams = _word_ngrams(hyp_words, n)
            ref_ngrams = _word_ngrams(ref_words, n)
            clipped_overlap += sum((hyp_ngrams & ref_ngrams).values())
            hyp_total += max(len(hyp_words) - n + 1, 0)
        # Additive smoothing: add 1 to both numerator and denominator so a
        # zero-overlap order contributes a small positive precision instead
        # of forcing the geometric mean to zero.
        precisions.append((clipped_overlap + 1) / (hyp_total + 1))

    for hyp, ref in zip(hypotheses, references, strict=True):
        hyp_len_total += len(hyp.split())
        ref_len_total += len(ref.split())

    log_precision_mean = sum(math.log(p) for p in precisions) / len(precisions)
    bp = _brevity_penalty(hyp_len_total, ref_len_total)
    return 100.0 * bp * math.exp(log_precision_mean)


# --- edit distance, WER/CER, TER (tech-spec v2 §6.1/§9) ------------------------


def _levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    """Classic O(len(a)*len(b)) edit distance (insertions, deletions,
    substitutions, each cost 1) over any sequence of hashable/comparable
    tokens — shared by word-level (WER/TER) and char-level (CER) callers."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    previous_row = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, token_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (0 if token_a == token_b else 1)
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
        previous_row = current_row
    return previous_row[-1]


def word_error_rate(hypothesis: str, reference: str) -> float:
    """WER = word-level edit distance / reference word count, as a fraction
    (not a percentage — tech-spec v2 §9 quotes WER bands like "25-40%" but
    this function returns the raw ratio; callers scale for display). A
    reference of zero words with a nonempty hypothesis scores 1.0 (every
    hypothesis word is an insertion, capped at 1.0 as this metric's ceiling
    is otherwise unbounded when the reference is empty); an empty/empty
    pair scores 0.0."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(hyp_words, ref_words) / len(ref_words)


def char_error_rate(hypothesis: str, reference: str) -> float:
    """CER = character-level edit distance / reference character count, as
    a fraction. Same empty-reference convention as word_error_rate."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(hypothesis, reference) / len(reference)


@dataclass(frozen=True, slots=True)
class WerCerDecomposed:
    """WER/CER scored against both reference layers (tech-spec v2 §9's
    "WER/CER against the diplomatic and the normalized reference, the
    difference isolating standardization"). The gap between the diplomatic
    and normalized scores is the standardization effect, not exposed as a
    separate field here since callers can always compute
    `wer_diplomatic - wer_normalized` themselves from the two recorded
    values without this type guessing which direction is "improvement" for
    them."""

    wer_diplomatic: float
    wer_normalized: float
    cer_diplomatic: float
    cer_normalized: float


def wer_cer_decomposed(
    hypothesis: str, reference_diplomatic: str, reference_normalized: str
) -> WerCerDecomposed:
    """Score `hypothesis` against both the diplomatic (verbatim) and
    normalized reference layers (tech-spec v2 §9 cWER/sWER), returning all
    four values so a caller can inspect the standardization gap directly."""
    return WerCerDecomposed(
        wer_diplomatic=word_error_rate(hypothesis, reference_diplomatic),
        wer_normalized=word_error_rate(hypothesis, reference_normalized),
        cer_diplomatic=char_error_rate(hypothesis, reference_diplomatic),
        cer_normalized=char_error_rate(hypothesis, reference_normalized),
    )


def token_error_rate(hypothesis: str, reference: str) -> float:
    """Token error rate for the `normalize` task (tech-spec v2 §6.1: "token
    error rate vs the gold normalized layer"): word-level edit distance
    over `reference`'s token count, as a fraction. Distinct function from
    `word_error_rate` (same underlying computation) so the normalize task's
    headline metric has its own name in call sites and reports, matching
    tech-spec v2 §6.1's per-task metric table where "token error rate" is
    the normalization row's named metric, not "WER" borrowed from speech."""
    return word_error_rate(hypothesis, reference)


# --- tokenizer fertility (tech-spec v2 §6.1: "tokens/word ... per candidate tokenizer") --


def tokenizer_fertility(texts: Sequence[str], tokenizer: Callable[[str], Sequence[object]]) -> float:
    """Mean tokens-per-word over `texts`, given a `tokenizer` callable
    (tech-spec v2 §6.1: "tokens/word on `frc` vs matched `fra` per
    candidate tokenizer" — the tokenizer itself is injected, never assumed,
    since this project compares candidate tokenizers rather than owning
    one). Word count is a whitespace split of the original text; token
    count is `len(tokenizer(text))`. A text with zero words contributes
    nothing to either total (skipped), so an all-empty `texts` raises
    rather than dividing by zero, since fertility is undefined with no
    words at all."""
    total_tokens = 0
    total_words = 0
    for text in texts:
        words = text.split()
        if not words:
            continue
        total_tokens += len(tokenizer(text))
        total_words += len(words)
    if total_words == 0:
        raise ValueError("tokenizer_fertility requires at least one non-empty text")
    return total_tokens / total_words


# --- Wilson CI and per-cell accuracy (tech-spec v2 §6.1/§6.3) ------------------


def wilson_confidence_interval(
    successes: int, n: int, *, z: float = _DEFAULT_WILSON_Z
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (tech-spec v2 §6.1
    "Wilson CIs", §6.3 "Wilson 95% lower bound"), returning (lower, upper)
    on the enum-independent proportion scale. `z` defaults to the two-sided
    95% critical value; a caller who needs the one-sided 95% lower bound
    used by §6.3's persistent-conflation rule reads only the returned
    lower bound with the same two-sided `z` (Wilson's own construction —
    not adjusted here for one- vs two-sided, since tech-spec v2 §6.3 does
    not specify a different `z` for its one-sided use). `n == 0` raises:
    a proportion over zero observations is undefined, not silently 0.0."""
    if n == 0:
        raise ValueError("wilson_confidence_interval requires n > 0")
    if not 0 <= successes <= n:
        raise ValueError("successes must satisfy 0 <= successes <= n")
    p_hat = successes / n
    z_sq = z * z
    denominator = 1 + z_sq / n
    center = p_hat + z_sq / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z_sq / (4 * n * n))
    # Endpoints are analytically exact at the extremes (successes == 0 → 0.0,
    # successes == n → 1.0); pin them so a libm rounding residue (CI on
    # Linux produced 6.9e-18 for the same formula in src/redteam) never leaks.
    lower = 0.0 if successes == 0 else max(0.0, (center - margin) / denominator)
    upper = 1.0 if successes == n else min(1.0, (center + margin) / denominator)
    return lower, upper


@dataclass(frozen=True, slots=True)
class CellAccuracy:
    """One diff-catalog cell's accuracy with its Wilson 95% CI (tech-spec
    v2 §6.1 "Diff-catalog coverage ... per-cell accuracy on probe sentences
    with Wilson CIs")."""

    cell_id: str
    n: int
    accuracy: float
    ci_lower: float
    ci_upper: float


def per_cell_accuracy_with_wilson_ci(
    cell_ids: Sequence[str], correct: Sequence[bool]
) -> dict[str, CellAccuracy]:
    """Group per-item pass/fail outcomes by diff-catalog `cell_id` (tech-
    spec v2 §4's cell ids, e.g. "VERB-001") and compute each cell's
    accuracy with a Wilson 95% CI (tech-spec v2 §6.1). `cell_ids` and
    `correct` are parallel sequences, one entry per probe item; must be the
    same length. A cell with zero items cannot appear (there is nothing to
    group by that id), so this never hits wilson_confidence_interval's
    n == 0 case."""
    if len(cell_ids) != len(correct):
        raise ValueError("cell_ids and correct must be the same length")
    totals: dict[str, int] = {}
    successes: dict[str, int] = {}
    for cell_id, is_correct in zip(cell_ids, correct, strict=True):
        totals[cell_id] = totals.get(cell_id, 0) + 1
        successes[cell_id] = successes.get(cell_id, 0) + (1 if is_correct else 0)

    result: dict[str, CellAccuracy] = {}
    for cell_id, n in totals.items():
        n_correct = successes[cell_id]
        lower, upper = wilson_confidence_interval(n_correct, n)
        result[cell_id] = CellAccuracy(
            cell_id=cell_id, n=n, accuracy=n_correct / n, ci_lower=lower, ci_upper=upper
        )
    return result


# --- paired bootstrap CI and the Q13 within_noise rule (ADR 0015) --------------


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Paired item-resampling bootstrap CI for the mean delta
    `scores_a[i] - scores_b[i]` (tech-spec v2 §3.2's decision rule / ADR
    0015's Q13 rule / tech-spec v2 §6.1 "deltas accepted only when the
    bootstrap CI excludes zero"): resample item indices with replacement
    `n_resamples` times (same resampled index set applied to both series,
    preserving the pairing), compute the mean delta each time, and return
    the `confidence`-level percentile interval (2.5th/97.5th percentile by
    default) as (lower, upper).

    `scores_a`/`scores_b` are equal-length per-item metric values (e.g.
    per-item chrF scores for two arms on the same locked eval set); `seed`
    is required, not defaulted, so every call site records the seed that
    produced its reported CI (ADR 0015 names 1,000 resamples but not a
    seed — this module's contract makes the caller choose and keep one,
    for reproducibility rather than silent nondeterminism)."""
    if len(scores_a) != len(scores_b):
        raise ValueError("paired_bootstrap_ci requires equal-length score series")
    if not scores_a:
        raise ValueError("paired_bootstrap_ci requires at least one paired item")

    deltas = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    n = len(deltas)
    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        resampled_means.append(sum(deltas[i] for i in indices) / n)
    resampled_means.sort()

    alpha = 1 - confidence
    lower_idx = max(0, min(n_resamples - 1, int(round((alpha / 2) * (n_resamples - 1)))))
    upper_idx = max(0, min(n_resamples - 1, int(round((1 - alpha / 2) * (n_resamples - 1)))))
    return resampled_means[lower_idx], resampled_means[upper_idx]


def within_noise(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int,
) -> bool:
    """The Q13 "within noise" rule (ADR 0015, tech-spec v2 §3.2's
    `decision_rule`): True when the 95% paired-bootstrap CI of the mean
    delta `scores_a - scores_b` contains zero (the comparison is "within
    noise" and, per ADR 0015's CARE tie-break, the arm named second/`b`
    wins — e.g. French-native as `scores_b`); False when the interval
    excludes zero (the arm with the higher score wins, decided by the
    caller from the raw scores — this function only answers the
    within-noise question, never which arm is "better", since that
    direction is CARE-grounds policy from ADR 0015, not a property of the
    numbers alone)."""
    lower, upper = paired_bootstrap_ci(scores_a, scores_b, n_resamples=n_resamples, seed=seed)
    return lower <= 0.0 <= upper


# --- MetricReport, compute_metrics, headline_metric (issue #23 row D2) --------


@dataclass(frozen=True, slots=True)
class MetricReport:
    """The typed result `compute_metrics` returns and
    `compute_acceptance_report` (src/eval/__init__.py) now accepts in place
    of a bare `dict[str, float]` — answering issue #23 row D2 ("which
    metric is the headline float") by naming one. `metrics` still carries
    every computed metric by name (unchanged shape for callers that want
    the full set); `headline_name`/`headline_value` pull out the one
    tech-spec v2 §6.1 names as the task's primary reported number.
    `task` records which per-task metric family produced this report, so a
    report is self-describing rather than requiring the caller to remember
    which `compute_metrics(..., task=...)` call produced it."""

    task: Task
    metrics: dict[str, float]
    headline_name: str
    headline_value: float
    per_cell: dict[str, CellAccuracy] = field(default_factory=dict)


def headline_metric(task: Task) -> str:
    """The tech-spec v2 §6.1 headline metric name per task family,
    answering issue #23 row D2 directly: MT → chrF++ (§6.1 "chrF++
    primary"); normalize → token error rate (§6.1 "token error rate vs the
    gold normalized layer"); contrast/judge → per-cell accuracy (§6.1
    "Diff-catalog coverage | per-cell accuracy on probe sentences ...").
    """
    if task == "mt":
        return "chrf_plus_plus"
    if task == "normalize":
        return "token_error_rate"
    if task in ("contrast", "judge"):
        return "per_cell_accuracy"
    raise ValueError(f"no headline metric defined for task {task!r}")


def compute_metrics(
    items: Sequence[_EvalItemLike],
    hypotheses: Sequence[str],
    task: Task,
    *,
    language: LanguageTag | None = None,
) -> MetricReport:
    """Compute the tech-spec v2 §6.1 metric set for one task family over
    `items` (each carrying `reference`/`reference_diplomatic` and, for
    contrast/judge, `diff_catalog_flags`-style cell ids via `cell_id`) and
    their `hypotheses` (parallel, same length, supplied at scoring time —
    this module never stores model output). This is the
    `compute_metrics: Callable[[list[EvalItem], LanguageTag], dict[str,
    float]]` seam's real implementation, adapted to return a `MetricReport`
    rather than a bare dict; `src/eval/__init__.py`'s
    `compute_acceptance_report` accepts either shape (a `MetricReport` or a
    plain dict) so a caller may bind this function directly.

    `task="mt"` computes chrF++ (headline), chrF, and corpus BLEU against
    `reference`. `task="normalize"` computes token error rate (headline)
    against `reference` (the normalized layer) and, when
    `reference_diplomatic` is present on every item, the full
    `wer_cer_decomposed` set averaged over items. `task in
    ("contrast","judge")` computes per-cell accuracy with Wilson CIs
    (headline: the corpus-wide accuracy) from each item's `cell_id` and
    whether its hypothesis exactly equals its `reference` (an exact-match
    judgment — tech-spec v2 §6.3's probe schema `expected: str |
    {accept:bool}` reduces to string equality for the `str` case, which is
    the only form this module scores; `{accept: bool}` judgments are the
    caller's own comparison, out of this function's scope since a bool
    "accept" probe has no reference string to compare against).

    `language` is accepted (matching the `compute_metrics` seam's
    signature in src/eval/__init__.py) but unused for any of this ticket's
    metrics, all of which are computed from the supplied text alone.
    """
    if len(items) != len(hypotheses):
        raise ValueError("compute_metrics requires items and hypotheses of equal length")
    if not items:
        raise ValueError("compute_metrics requires at least one item")

    if task == "mt":
        return _compute_mt_metrics(items, hypotheses)
    if task == "normalize":
        return _compute_normalize_metrics(items, hypotheses)
    if task in ("contrast", "judge"):
        return _compute_cell_accuracy_metrics(items, hypotheses, task)
    raise ValueError(f"unsupported task {task!r}")


def _compute_mt_metrics(items: Sequence[_EvalItemLike], hypotheses: Sequence[str]) -> MetricReport:
    references = [_require_reference(item) for item in items]
    per_item_chrf_pp = [chrf_plus_plus(hyp, ref) for hyp, ref in zip(hypotheses, references, strict=True)]
    per_item_chrf = [chrf(hyp, ref) for hyp, ref in zip(hypotheses, references, strict=True)]
    metrics = {
        "chrf_plus_plus": sum(per_item_chrf_pp) / len(per_item_chrf_pp),
        "chrf": sum(per_item_chrf) / len(per_item_chrf),
        "bleu": corpus_bleu(list(hypotheses), references),
    }
    return MetricReport(
        task="mt",
        metrics=metrics,
        headline_name=headline_metric("mt"),
        headline_value=metrics["chrf_plus_plus"],
    )


def _compute_normalize_metrics(items: Sequence[_EvalItemLike], hypotheses: Sequence[str]) -> MetricReport:
    references = [_require_reference(item) for item in items]
    per_item_ter = [token_error_rate(hyp, ref) for hyp, ref in zip(hypotheses, references, strict=True)]
    metrics = {"token_error_rate": sum(per_item_ter) / len(per_item_ter)}

    diplomatic_refs = [item.reference_diplomatic for item in items]
    if all(ref is not None for ref in diplomatic_refs):
        decomposed = [
            wer_cer_decomposed(hyp, diplomatic, ref)
            for hyp, diplomatic, ref in zip(hypotheses, diplomatic_refs, references, strict=True)
            if diplomatic is not None
        ]
        metrics["wer_diplomatic"] = sum(d.wer_diplomatic for d in decomposed) / len(decomposed)
        metrics["wer_normalized"] = sum(d.wer_normalized for d in decomposed) / len(decomposed)
        metrics["cer_diplomatic"] = sum(d.cer_diplomatic for d in decomposed) / len(decomposed)
        metrics["cer_normalized"] = sum(d.cer_normalized for d in decomposed) / len(decomposed)

    return MetricReport(
        task="normalize",
        metrics=metrics,
        headline_name=headline_metric("normalize"),
        headline_value=metrics["token_error_rate"],
    )


def _compute_cell_accuracy_metrics(
    items: Sequence[_EvalItemLike], hypotheses: Sequence[str], task: Task
) -> MetricReport:
    cell_ids = [_require_cell_id(item) for item in items]
    references = [_require_reference(item) for item in items]
    correct = [hyp == ref for hyp, ref in zip(hypotheses, references, strict=True)]

    per_cell = per_cell_accuracy_with_wilson_ci(cell_ids, correct)
    overall_accuracy = sum(correct) / len(correct)
    metrics = {"per_cell_accuracy": overall_accuracy}
    metrics.update({f"accuracy[{cell_id}]": cell.accuracy for cell_id, cell in per_cell.items()})

    return MetricReport(
        task=task,
        metrics=metrics,
        headline_name=headline_metric(task),
        headline_value=overall_accuracy,
        per_cell=per_cell,
    )


def _require_reference(item: _EvalItemLike) -> str:
    if item.reference is None:
        raise ValueError(f"item {item.item_id!r} has no reference; cannot score this task")
    return item.reference


def _require_cell_id(item: _EvalItemLike) -> str:
    cell_id = getattr(item, "cell_id", None)
    if cell_id is None:
        raise ValueError(f"item {item.item_id!r} has no cell_id; cannot compute per-cell accuracy")
    return str(cell_id)


class _EvalItemLike(Protocol):
    """Structural (`typing.Protocol`) stand-in documenting the attributes
    `compute_metrics` reads off each item: `item_id`, `reference`,
    `reference_diplomatic`, and (for contrast/judge tasks) `cell_id`. A
    `Protocol`, not a concrete base class, so `EvalItem` in
    `src/eval/__init__.py` satisfies this shape structurally without
    inheriting from it — avoiding an import cycle (`metrics.py` would
    otherwise need to import `EvalItem` from `__init__.py`, which imports
    `MetricReport` from `metrics.py`) and keeping `compute_metrics` usable
    in tests with any object exposing these attributes, per this ticket's
    precedent check that no production call site outside
    `src/eval`/`tests/test_eval.py` fixes a signature that would block
    this.

    Declared as read-only `@property` members, not plain attribute
    annotations (issue #43 finding): a plain `item_id: str`-style Protocol
    member requires a *settable* variable for structural conformance, which
    `EvalItem` — a `frozen=True, slots=True` dataclass — can never offer,
    so `compute_metrics(list[EvalItem], ...)` failed mypy's Protocol check
    even though `EvalItem` satisfies this shape at runtime. A read-only
    property member only requires a readable attribute, which a frozen
    dataclass field already is, so `Sequence[EvalItem]` is now directly
    assignable to `Sequence[_EvalItemLike]` with no cast at any call site."""

    @property
    def item_id(self) -> str: ...

    @property
    def reference(self) -> str | None: ...

    @property
    def reference_diplomatic(self) -> str | None: ...

    @property
    def cell_id(self) -> str | None: ...
