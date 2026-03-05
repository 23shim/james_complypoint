"""Tests for text tokenisation."""

from classification.tokeniser import all_ngrams, ngrams, tokenise


class TestTokenise:

    def test_simple_text(self):
        assert tokenise("Fire safety") == ["fire", "safety"]

    def test_preserves_ampersand(self):
        assert tokenise("Health & Safety") == ["health", "&", "safety"]

    def test_underscores_split(self):
        assert tokenise("my_document_v2") == ["my", "document", "v2"]

    def test_strips_parentheses(self):
        assert tokenise("EICR (10938532)") == ["eicr", "10938532"]

    def test_strips_brackets(self):
        assert tokenise("Drawings [Latest]") == ["drawings", "latest"]

    def test_lowercase(self):
        assert tokenise("FINANCE") == ["finance"]

    def test_empty_string(self):
        assert tokenise("") == []

    def test_real_folder_fire_safety(self):
        result = tokenise("Fire safety & risk assessments")
        assert result == ["fire", "safety", "&", "risk", "assessments"]

    def test_real_folder_handover(self):
        result = tokenise("Handover")
        assert result == ["handover"]

    def test_real_folder_with_dates(self):
        result = tokenise("Drawings (Latest) Sept 2016")
        assert result == ["drawings", "latest", "sept", "2016"]

    def test_real_filename_eicr(self):
        result = tokenise("1 Fuggles Close EICR (10938532)")
        assert result == ["1", "fuggles", "close", "eicr", "10938532"]

    def test_hyphenated_text(self):
        result = tokenise("EDRM-Parsonage Place")
        assert result == ["edrm-parsonage", "place"]

    def test_multiple_spaces(self):
        result = tokenise("  lots   of   spaces  ")
        assert result == ["lots", "of", "spaces"]


class TestNgrams:

    def test_bigrams(self):
        tokens = ["fire", "risk", "assessment"]
        assert ngrams(tokens, 2) == ["fire risk", "risk assessment"]

    def test_trigrams(self):
        tokens = ["fire", "risk", "assessment"]
        assert ngrams(tokens, 3) == ["fire risk assessment"]

    def test_too_few_tokens(self):
        assert ngrams(["fire"], 2) == []

    def test_empty_tokens(self):
        assert ngrams([], 2) == []


class TestAllNgrams:

    def test_returns_longest_first(self):
        tokens = ["health", "and", "safety"]
        result = all_ngrams(tokens, max_n=3)
        assert result[0] == "health and safety"

    def test_includes_single_tokens(self):
        tokens = ["fire", "safety"]
        result = all_ngrams(tokens, max_n=2)
        assert "fire safety" in result
        assert "fire" in result
        assert "safety" in result
