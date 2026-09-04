"""Tests for src/lid/ — the interim rule-based LID classifier baseline.

Tests only external behavior: calling classify() and asserting on the
returned LIDResult. Never asserts on which specific rule/heuristic fired.
"""

import socket
from unittest.mock import patch

from data_contract import LanguageTag
from lid import CodeSwitchSpan, LIDResult, classify


def test_frc_marked_text_returns_frc_with_high_confidence() -> None:
    result = classify("Asteur ouais pis on va faire ça lâ-bas.")
    assert result.language_tag == "frc"
    assert result.confidence >= 0.5


def test_lou_marked_text_returns_lou_with_high_confidence() -> None:
    result = classify("Nonm-lá ka kalkile pou kouri vini.")
    assert result.language_tag == "lou"
    assert result.confidence >= 0.5


def test_ambiguous_text_returns_unknown_with_low_confidence() -> None:
    result = classify("xyzzy plugh qwerty asdf")
    assert result.language_tag == "unknown"
    assert result.confidence < 0.5
    assert result.spans == ()


def test_code_switched_text_returns_mixed_with_spans() -> None:
    text = "Asteur pis the going with you"
    result = classify(text)
    assert result.language_tag == "mixed"
    # The two code-switched segments must be reported as two merged spans
    # (not one per matched token) whose offsets correctly bracket each
    # contiguous same-language run in the original text.
    assert result.spans == (
        CodeSwitchSpan(start=0, end=10, language_tag="frc"),
        CodeSwitchSpan(start=11, end=29, language_tag="eng"),
    )
    assert text[0:10] == "Asteur pis"
    assert text[11:29] == "the going with you"


def test_language_hint_changes_verdict_when_evidence_is_weak() -> None:
    # A single frc marker is "weak" evidence (below _HINT_MARKER_COUNT_THRESHOLD):
    # the hint may break the tie toward a different tag entirely.
    text = "Asteur on va faire ça."
    unhinted = classify(text)
    hinted = classify(text, language_hint="lou")
    assert unhinted.language_tag == "frc"
    assert hinted.language_tag == "lou"
    assert hinted != unhinted


def test_language_hint_lifts_confidence_when_it_agrees_with_weak_evidence() -> None:
    text = "Asteur on va faire ça."
    unhinted = classify(text)
    hinted = classify(text, language_hint="frc")
    assert unhinted.language_tag == hinted.language_tag == "frc"
    assert hinted.confidence > unhinted.confidence


def test_language_hint_does_not_override_strong_contrary_evidence() -> None:
    # Text has multiple eng markers (strong evidence); a lou hint must not
    # force a lou verdict, and must not even change the confidence.
    text = "The going with you and the have"
    unhinted = classify(text)
    hinted = classify(text, language_hint="lou")
    assert hinted.language_tag == "eng"
    assert hinted == unhinted


def test_standard_french_text_returns_fra() -> None:
    result = classify("Je ne sais pas ce que vous voulez dire, monsieur, cependant.")
    assert result.language_tag == "fra"


def test_haitian_creole_text_returns_hat() -> None:
    result = classify("Mwen renmen anpil kijan yo rele lakay kounye a.")
    assert result.language_tag == "hat"


def test_spa_marked_text_returns_spa_with_high_confidence() -> None:
    result = classify("Pero también entonces porque señor ustedes.")
    assert result.language_tag == "spa"
    assert result.confidence >= 0.5


def test_eng_marked_text_carries_eng_dialect_unknown() -> None:
    result = classify("The going with you and the have.")
    assert result.language_tag == "eng"
    assert result.eng_dialect == "unknown"


def test_non_eng_result_has_eng_dialect_none() -> None:
    frc = classify("Asteur ouais pis on va faire ça lâ-bas.")
    lou = classify("Nonm-lá ka kalkile pou kouri vini.")
    fra = classify("Je ne sais pas ce que vous voulez dire, monsieur, cependant.")
    hat = classify("Mwen renmen anpil kijan yo rele lakay kounye a.")
    spa = classify("Pero también entonces porque señor ustedes.")
    for result in (frc, lou, fra, hat, spa):
        assert result.eng_dialect is None


def test_spa_markers_do_not_overlap_existing_marker_sets() -> None:
    """Set-intersection guard across all six marker frozensets: accidental
    duplication would introduce false positives across languages."""
    from lid import _ENG_MARKERS, _FRA_MARKERS, _FRC_MARKERS, _HAT_MARKERS, _LOU_MARKERS, _SPA_MARKERS

    marker_sets = {
        "frc": _FRC_MARKERS,
        "lou": _LOU_MARKERS,
        "eng": _ENG_MARKERS,
        "fra": _FRA_MARKERS,
        "hat": _HAT_MARKERS,
        "spa": _SPA_MARKERS,
    }
    for name_a, markers_a in marker_sets.items():
        for name_b, markers_b in marker_sets.items():
            if name_a >= name_b:
                continue
            assert not (markers_a & markers_b), f"{name_a} and {name_b} marker sets overlap"


def test_language_hint_spa_works_through_existing_weak_evidence_branch() -> None:
    # A single frc marker is "weak" evidence: the hint may break the tie
    # toward a different tag entirely, including the new spa tag.
    text = "Asteur on va faire ça."
    unhinted = classify(text)
    hinted = classify(text, language_hint="spa")
    assert unhinted.language_tag == "frc"
    assert hinted.language_tag == "spa"
    assert hinted != unhinted


def test_classify_is_deterministic() -> None:
    text = "Asteur ouais pis on va faire ça lâ-bas."
    first = classify(text)
    second = classify(text)
    assert first == second


def test_classify_touches_no_filesystem_or_network() -> None:
    """Test-double check (same technique as data_contract.py's purity test):
    patch open() and socket.connect to raise, then run classify() across
    every representative case. If it were to touch disk or network, this
    test fails loudly instead of by omission."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("classify() must not touch the filesystem or network")

    cases = [
        "Asteur ouais pis on va faire ça lâ-bas.",
        "Nonm-lá ka kalkile pou kouri vini.",
        "xyzzy plugh qwerty asdf",
        "Asteur pis the going with you",
        "Je ne sais pas ce que vous voulez dire, monsieur, cependant.",
        "Mwen renmen anpil kijan yo rele lakay kounye a.",
    ]

    with (
        patch("builtins.open", side_effect=_boom),
        patch.object(socket.socket, "connect", side_effect=_boom),
    ):
        for text in cases:
            classify(text)


def test_lid_result_language_tag_shares_data_contract_enum() -> None:
    """LIDResult.language_tag must be data_contract's LanguageTag, not a
    second, independently-defined enum that happens to look alike."""
    result = classify("Asteur ouais pis on va faire ça.")
    tag: LanguageTag = result.language_tag  # mypy-checked assignment
    assert tag in ("lou", "frc", "fra", "hat", "eng", "spa", "mixed", "unknown")


def test_code_switch_span_is_a_named_type() -> None:
    result = classify("Asteur pis the going with you")
    for span in result.spans:
        assert isinstance(span, CodeSwitchSpan)


def test_lid_result_is_a_named_type() -> None:
    result = classify("Asteur ouais pis on va faire ça.")
    assert isinstance(result, LIDResult)
