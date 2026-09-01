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

from data_contract import LanguageTag

# Placeholder marker sets (engineering stand-in, not linguist-curated — see
# module docstring). Each set is a short list of tokens whose presence is
# treated as evidence for that language. Matching is case-insensitive,
# whole-word.
_FRC_MARKERS = frozenset({"asteur", "lâ-bas", "faisait", "c'est-tu", "ouais", "pis"})
_LOU_MARKERS = frozenset({"nonm", "fanm", "kalkile", "kouri", "vini", "lapèl"})
_ENG_MARKERS = frozenset({"the", "and", "you", "with", "have", "going"})

# Fixed heuristic confidence levels — an interim baseline has no calibrated
# probability model, so these are not learned scores.
_HIGH_CONFIDENCE = 0.75
_LOW_CONFIDENCE = 0.2


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
    """The verdict classify() returns: a dominant language tag, a fixed
    heuristic confidence level, and — only when language_tag == "mixed" —
    the code-switch spans that produced that verdict."""

    language_tag: LanguageTag
    confidence: float
    spans: tuple[CodeSwitchSpan, ...]


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


def classify(text: str, *, language_hint: LanguageTag | None = None) -> LIDResult:
    """Classify `text`'s dominant language, per the interim baseline described
    in this module's docstring. Pure function: no I/O, no side effects,
    deterministic for a given (text, language_hint) pair.

    `language_hint` may reinforce a tag already supported by marker
    evidence, but never overrides strong contrary evidence (e.g. a `lou`
    hint against text with only `eng` markers still returns `eng`).
    """
    tokens = _tokenize_with_offsets(text)

    frc_spans = _find_marker_spans(tokens, _FRC_MARKERS, "frc")
    lou_spans = _find_marker_spans(tokens, _LOU_MARKERS, "lou")
    eng_spans = _find_marker_spans(tokens, _ENG_MARKERS, "eng")

    hits: dict[LanguageTag, list[CodeSwitchSpan]] = {}
    if frc_spans:
        hits["frc"] = frc_spans
    if lou_spans:
        hits["lou"] = lou_spans
    if eng_spans:
        hits["eng"] = eng_spans

    if not hits:
        return LIDResult(language_tag="unknown", confidence=_LOW_CONFIDENCE, spans=())

    if len(hits) > 1:
        # Real evidence of two or more languages' markers co-occurring —
        # the only condition under which classify() reports mixed; every
        # reported span boundary is backed by an actual marker match.
        all_spans = tuple(sorted((s for spans in hits.values() for s in spans), key=lambda s: s.start))
        return LIDResult(language_tag="mixed", confidence=_HIGH_CONFIDENCE, spans=all_spans)

    (dominant_tag,) = hits.keys()

    if language_hint is not None and language_hint in hits:
        dominant_tag = language_hint

    return LIDResult(language_tag=dominant_tag, confidence=_HIGH_CONFIDENCE, spans=())
