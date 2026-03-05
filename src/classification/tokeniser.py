"""
Text tokenisation for classification.

Normalises text, splits into tokens, and produces n-grams
for multi-word phrase matching against dictionaries.

Design choices:
- Preserves '&' and '-' as they appear in real folder names
  like "Health & Safety" and "O&M Manual"
- Strips parentheses/brackets — "(Updated Sept 17)" becomes
  just tokens, not noise
- Everything lowercased for case-insensitive matching
"""

import re

# Pattern to split text into tokens.
# Splits on whitespace, underscores, and common delimiters,
# but preserves '&' and '-' as standalone tokens.
_SPLIT_PATTERN = re.compile(r"[_\s]+")

# Characters to strip from individual tokens
_STRIP_CHARS = "()[]{}.,;:'\"!?#"


def tokenise(text: str) -> list[str]:
    """Tokenise a text string into lowercase tokens.

    Args:
        text: Raw text (folder segment name, filename stem, etc.)

    Returns:
        List of lowercase tokens with punctuation stripped.

    Examples:
        >>> tokenise("Fire safety & risk assessments")
        ['fire', 'safety', '&', 'risk', 'assessments']
        >>> tokenise("EICR (10938532)")
        ['eicr', '10938532']
        >>> tokenise("my_document_v2")
        ['my', 'document', 'v2']
    """
    if not text:
        return []

    text = text.lower()
    raw_tokens = _SPLIT_PATTERN.split(text)

    tokens = []
    for raw in raw_tokens:
        cleaned = raw.strip(_STRIP_CHARS)
        if cleaned:
            tokens.append(cleaned)

    return tokens


def ngrams(tokens: list[str], n: int) -> list[str]:
    """Produce n-grams from a token list.

    Args:
        tokens: List of tokens from tokenise().
        n: N-gram size (2 for bigrams, 3 for trigrams, etc.)

    Returns:
        List of space-joined n-grams.

    Examples:
        >>> ngrams(["fire", "risk", "assessment"], 2)
        ['fire risk', 'risk assessment']
        >>> ngrams(["fire", "risk", "assessment"], 3)
        ['fire risk assessment']
    """
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def all_ngrams(tokens: list[str], max_n: int = 4) -> list[str]:
    """Produce all n-grams from 1 to max_n.

    Returns single tokens plus all bigrams, trigrams, etc.
    Useful for matching multi-word dictionary entries like
    "fire risk assessment" or "health and safety".

    Args:
        tokens: List of tokens from tokenise().
        max_n: Maximum n-gram size.

    Returns:
        List of all n-gram phrases, longest first (for greedy matching).
    """
    result = []
    for n in range(min(max_n, len(tokens)), 0, -1):
        result.extend(ngrams(tokens, n))
    return result
