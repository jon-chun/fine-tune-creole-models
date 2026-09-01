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
    assert len(result.spans) >= 2
    tags_seen = {span.language_tag for span in result.spans}
    assert "frc" in tags_seen
    assert "eng" in tags_seen
    # Every span's slice of the original text must actually be the token that matched.
    for span in result.spans:
        assert text[span.start : span.end]


def test_language_hint_biases_toward_agreeing_evidence() -> None:
    # Sole detected language is frc; a matching hint is reinforced, not overridden.
    result = classify("Asteur ouais pis on va faire ça.", language_hint="frc")
    assert result.language_tag == "frc"


def test_language_hint_does_not_override_strong_contrary_evidence() -> None:
    # Text has only eng markers; a lou hint must not force a lou verdict.
    result = classify("The going with you and the have", language_hint="lou")
    assert result.language_tag == "eng"


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
    assert tag in ("lou", "frc", "fra", "hat", "eng", "mixed", "unknown")


def test_code_switch_span_is_a_named_type() -> None:
    result = classify("Asteur pis the going with you")
    for span in result.spans:
        assert isinstance(span, CodeSwitchSpan)


def test_lid_result_is_a_named_type() -> None:
    result = classify("Asteur ouais pis on va faire ça.")
    assert isinstance(result, LIDResult)
