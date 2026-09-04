"""Language-ID interface: LIDResult, CodeSwitchSpan, and classify().

INTERIM BASELINE — not the tech-spec §3.1 classifier. tech-spec §3.1 calls
for a fastText- or CamemBERT-based classifier trained over a seed corpus,
evaluated on macro-F1 and targeted confusion-pair F1 against a Layer-B gold
set (ARD §6). None of that exists yet: no seed corpus, no gold set, no
trained model. This module exists so that every downstream consumer
(fine-tune-cajun-preprocess.py, coreset selection, the eval harness) can be
built and tested against a stable `classify(text) -> LIDResult` contract
today, instead of blocking on that research question landing first.

This module makes NO accuracy claim (no macro-F1, no confusion-pair F1) —
those require the real classifier and a gold set. Swapping this baseline
for the real one later is a drop-in replacement behind the same
`classify()` signature and `LIDResult` return type; no caller changes.

The baseline's tagging logic is a small, auditable set of marker-wordlist
heuristics — an engineering placeholder, not a linguist-reviewed wordlist
(that is future work once the linguist role is engaged, mirroring the
`variant_rules.yaml` gap recorded in the repo's CONTEXT.md). Its weaknesses
are real and expected: text outside its small marker sets comes back
`unknown` rather than a guessed default.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_contract import EngDialect, LanguageTag

# Placeholder marker sets (engineering stand-in, not linguist-curated — see
# module docstring). Each set is a short list of tokens whose presence is
# treated as evidence for that language. Matching is case-insensitive,
# whole-word.
_FRC_MARKERS = frozenset({"asteur", "lâ-bas", "faisait", "c'est-tu", "ouais", "pis"})
_LOU_MARKERS = frozenset({"nonm", "fanm", "kalkile", "kouri", "vini", "lapèl"})
_ENG_MARKERS = frozenset({"the", "and", "you", "with", "have", "going"})

# Placeholder marker sets (engineering stand-in, not linguist-curated — see
# module docstring). draft-unreviewed, same caveat as configs/diff_catalog/
# YAML files' meta.status: transcribed from general knowledge of Standard
# French / Haitian Creole closed-class function words, not attested against
# a corpus. Kept deliberately small and non-overlapping with the frc/lou/eng
# sets above so fra/hat become reachable outputs without new false positives
# on existing tests.
_FRA_MARKERS = frozenset({"aujourd'hui", "beaucoup", "cependant", "néanmoins", "voulez", "monsieur"})
_HAT_MARKERS = frozenset({"mwen", "yo", "kijan", "kounye", "lakay", "anpil"})

# Placeholder marker set (engineering stand-in, not linguist-curated — see
# module docstring), added for the v2 language-tag set (tech-spec v2 §3.1:
# "A multi-way classifier over {lou, frc, fra, hat, eng, spa, mixed,
# unknown}"; MIG-01h). draft-unreviewed, same caveat as the fra/hat sets
# above: transcribed from general knowledge of Spanish closed-class function
# words, not attested against a corpus. Kept deliberately small and
# non-overlapping with the frc/lou/eng/fra/hat sets above so spa becomes a
# reachable output without new false positives on existing tests (guarded by
# test_spa_markers_do_not_overlap_existing_marker_sets).
_SPA_MARKERS = frozenset({"pero", "también", "porque", "entonces", "señor", "ustedes"})

# Fixed heuristic confidence levels — an interim baseline has no calibrated
# probability model, so these are not learned scores.
_HIGH_CONFIDENCE = 0.75
_LOW_CONFIDENCE = 0.2

# Confidence for a sole-language verdict backed by only one matching marker
# token. Below _HINT_MARKER_COUNT_THRESHOLD, i.e. "weak" (see below).
_WEAK_MARKER_CONFIDENCE = 0.4

# A sole detected language's evidence counts as "weak" when its matched-marker
# count is below this threshold — the only condition under which a
# caller-supplied language_hint may affect the verdict (deliverable #1 of
# issue #17): it may then break the tie or lift confidence, but strong
# (>= threshold) evidence always wins over the hint, so a hint can never
# override real contrary marker evidence.
_HINT_MARKER_COUNT_THRESHOLD = 2


@dataclass(frozen=True, slots=True)
class CodeSwitchSpan:
    """One contiguous run of text attributed to a single language within a
    `mixed`-tagged result. `start`/`end` are character offsets into the
    original input text (end-exclusive, like Python slicing)."""

    start: int
    end: int
    language_tag: LanguageTag


@dataclass(frozen=True, slots=True)
class LIDResult:
    """The verdict classify() returns: the winning language tag (never a
    collapsed dominant-language label — see CONTEXT.md's Language-ID
    taxonomy entry; "mixed" is reported as its own tag with spans, not
    resolved into one language), a fixed heuristic confidence level, and —
    only when language_tag == "mixed" — the code-switch spans that produced
    that verdict.

    `eng_dialect` (tech-spec v2 §3.1; MIG-01h) is meaningful only when
    `language_tag == "eng"` — mirrors the existing "meaningful only when..."
    pattern this repo already uses for optional sub-fields (e.g.
    `DataItem.community_review_signed_off`, `src/data_contract.py`). It is
    `None` for every other `language_tag`. This interim baseline has no real
    dialect classifier: whenever `language_tag == "eng"`, `eng_dialect` is
    always `"unknown"` rather than a guessed `aae`/`other` verdict, per this
    module's "makes NO accuracy claim" posture — see module docstring."""

    language_tag: LanguageTag
    confidence: float
    spans: tuple[CodeSwitchSpan, ...]
    eng_dialect: EngDialect | None = None


def _tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split on whitespace, keeping each token's original character offsets."""
    tokens: list[tuple[str, int, int]] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                tokens.append((text[start:i], start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        tokens.append((text[start:], start, len(text)))
    return tokens


def _find_marker_spans(
    tokens: list[tuple[str, int, int]], markers: frozenset[str], tag: LanguageTag
) -> list[CodeSwitchSpan]:
    spans: list[CodeSwitchSpan] = []
    for token, start, end in tokens:
        stripped = token.strip(".,!?;:\"'").lower()
        if stripped in markers:
            spans.append(CodeSwitchSpan(start=start, end=end, language_tag=tag))
    return spans


def _merge_adjacent_same_tag_spans(spans: tuple[CodeSwitchSpan, ...]) -> tuple[CodeSwitchSpan, ...]:
    """Merge per-token marker spans into contiguous same-language segments.

    `_find_marker_spans` yields one span per matched token, so a run of
    consecutive matched tokens sharing a language_tag (e.g. "the going with
    you", four separate eng-marker spans) shows up as several separate,
    adjacent spans. A `mixed` verdict's spans should bracket segments, not
    individual matched tokens: given `spans` already sorted by `start` and
    covering every matched token in token order, this collapses each run of
    immediately-consecutive same-tag spans into one span running from the
    first token's start to the last token's end. A run breaks as soon as
    the next matched token has a different tag, even if unmatched filler
    tokens sit between two same-tag matches — those filler tokens' actual
    language is unknown to this heuristic, so bridging over them would be
    an unsupported claim.
    """
    if not spans:
        return ()

    merged: list[CodeSwitchSpan] = []
    current_tag = spans[0].language_tag
    current_start = spans[0].start
    current_end = spans[0].end

    for span in spans[1:]:
        if span.language_tag == current_tag:
            current_end = span.end
        else:
            merged.append(CodeSwitchSpan(start=current_start, end=current_end, language_tag=current_tag))
            current_tag = span.language_tag
            current_start = span.start
            current_end = span.end

    merged.append(CodeSwitchSpan(start=current_start, end=current_end, language_tag=current_tag))
    return tuple(merged)


def classify(text: str, *, language_hint: LanguageTag | None = None) -> LIDResult:
    """Classify `text`'s language tag, per the interim baseline described
    in this module's docstring. Pure function: no I/O, no side effects,
    deterministic for a given (text, language_hint) pair.

    Evidence for the sole detected language is "weak" when fewer than
    `_HINT_MARKER_COUNT_THRESHOLD` marker tokens matched. Only then may
    `language_hint` affect the verdict: if the hint names a *different* tag
    than the weak evidence suggests, the hint's tag wins (it resolves the
    ambiguity); if the hint agrees with the weak evidence, confidence is
    lifted since the two now corroborate each other. Either way the result
    is reported at `_HIGH_CONFIDENCE`. Strong evidence (>= threshold
    matching markers) is never overridden by a hint — e.g. a `lou` hint
    against text with multiple `eng` markers still returns `eng` at full
    confidence.
    """
    tokens = _tokenize_with_offsets(text)

    frc_spans = _find_marker_spans(tokens, _FRC_MARKERS, "frc")
    lou_spans = _find_marker_spans(tokens, _LOU_MARKERS, "lou")
    fra_spans = _find_marker_spans(tokens, _FRA_MARKERS, "fra")
    hat_spans = _find_marker_spans(tokens, _HAT_MARKERS, "hat")
    eng_spans = _find_marker_spans(tokens, _ENG_MARKERS, "eng")
    spa_spans = _find_marker_spans(tokens, _SPA_MARKERS, "spa")

    hits: dict[LanguageTag, list[CodeSwitchSpan]] = {}
    if frc_spans:
        hits["frc"] = frc_spans
    if lou_spans:
        hits["lou"] = lou_spans
    if fra_spans:
        hits["fra"] = fra_spans
    if hat_spans:
        hits["hat"] = hat_spans
    if eng_spans:
        hits["eng"] = eng_spans
    if spa_spans:
        hits["spa"] = spa_spans

    if not hits:
        return LIDResult(language_tag="unknown", confidence=_LOW_CONFIDENCE, spans=(), eng_dialect=None)

    if len(hits) > 1:
        # Real evidence of two or more languages' markers co-occurring —
        # the only condition under which classify() reports mixed; every
        # reported span boundary is backed by an actual marker match, and
        # adjacent same-tag matches are merged into one segment span (see
        # _merge_adjacent_same_tag_spans) rather than reported per-token.
        all_spans = tuple(sorted((s for spans in hits.values() for s in spans), key=lambda s: s.start))
        merged_spans = _merge_adjacent_same_tag_spans(all_spans)
        return LIDResult(language_tag="mixed", confidence=_HIGH_CONFIDENCE, spans=merged_spans, eng_dialect=None)

    (sole_tag,) = hits.keys()
    is_weak_evidence = len(hits[sole_tag]) < _HINT_MARKER_COUNT_THRESHOLD

    if language_hint is not None and is_weak_evidence:
        # Weak evidence: the hint may break the tie (adopt the hint's tag)
        # or corroborate the existing weak tag — either way confidence is
        # no longer merely weak, since a second signal now agrees with it.
        hinted_eng_dialect: EngDialect | None = "unknown" if language_hint == "eng" else None
        return LIDResult(
            language_tag=language_hint, confidence=_HIGH_CONFIDENCE, spans=(), eng_dialect=hinted_eng_dialect
        )

    confidence = _WEAK_MARKER_CONFIDENCE if is_weak_evidence else _HIGH_CONFIDENCE
    # eng_dialect is meaningful only when language_tag == "eng" (see LIDResult
    # docstring); this interim baseline has no real dialect classifier, so it
    # is always "unknown" rather than a guessed aae/other verdict (tech-spec
    # v2 §3.1; MIG-01h — makes no accuracy claim, per module docstring).
    sole_eng_dialect: EngDialect | None = "unknown" if sole_tag == "eng" else None
    return LIDResult(language_tag=sole_tag, confidence=confidence, spans=(), eng_dialect=sole_eng_dialect)
