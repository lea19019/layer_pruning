"""Tests for src/evaluation/translate.py.

Focuses on the _extract_translation post-processing that prevents
hallucinated content after the first translation.
"""

import pytest

from src.evaluation.translate import _extract_translation, STOP_STRINGS


class TestExtractTranslation:
    def test_clean_translation_unchanged(self):
        text = "Dies ist eine Übersetzung."
        assert _extract_translation(text) == text

    def test_truncates_at_newline_czech(self):
        text = "Dies ist gut.\nCzech: Neco dalsiho.\nGerman: Etwas anderes."
        assert _extract_translation(text) == "Dies ist gut."

    def test_truncates_at_src_lang_name(self):
        """Stop at the configured SRC_LANG_NAME (Czech)."""
        text = "Guten Tag.\nCzech: Ahoj."
        assert _extract_translation(text) == "Guten Tag."

    def test_truncates_at_double_newline(self):
        text = "Erste Zeile.\n\nZweite Absatz mit Halluzination."
        assert _extract_translation(text) == "Erste Zeile."

    def test_strips_whitespace(self):
        text = "  Hallo Welt  "
        assert _extract_translation(text) == "Hallo Welt"

    def test_empty_string(self):
        assert _extract_translation("") == ""

    def test_only_stop_string(self):
        assert _extract_translation("\nCzech:") == ""

    def test_multiline_hallucination(self):
        """Realistic example: model generates translation then keeps going."""
        text = (
            "Der Nationalismus ist zurückgegangen."
            "\nCzech: Nacionalismus ustoupil."
            "\nGerman: Der Nationalismus ist zurückgegangen."
            "\n\n(Translation complete)"
        )
        assert _extract_translation(text) == "Der Nationalismus ist zurückgegangen."

    def test_truncates_at_german_label(self):
        text = "Übersetzung hier.\nGerman: Nochmal dasselbe."
        assert _extract_translation(text) == "Übersetzung hier."

    def test_truncates_at_translation_label(self):
        text = "Guten Tag.\nTranslation: Good day."
        assert _extract_translation(text) == "Guten Tag."

    def test_truncates_at_translation_paren(self):
        text = "Guten Tag.\n(Translation complete)"
        assert _extract_translation(text) == "Guten Tag."

    def test_preserves_single_newline_in_translation(self):
        """A single newline NOT followed by a stop pattern should be kept
        only if it doesn't match any stop string."""
        text = "Zeile eins.\nZeile zwei."
        # This does not match any stop string, so it should be preserved
        assert _extract_translation(text) == "Zeile eins.\nZeile zwei."

    def test_stop_strings_are_defined(self):
        assert len(STOP_STRINGS) > 0
        assert any("Czech" in s for s in STOP_STRINGS)
        assert "\n\n" in STOP_STRINGS
