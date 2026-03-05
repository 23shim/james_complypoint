"""
Character-level similarity utilities for clustering.

Provides character trigrams, Jaccard similarity, and Union-Find
for single-linkage clustering. Operates at the character level,
complementing tokeniser.py which operates at the word level.

Used by entity_cluster.py.
"""

from __future__ import annotations

from collections import defaultdict


# ---- Union-Find ----


class UnionFind:
    """Disjoint Set Union with path compression and union by rank.

    Used for single-linkage clustering: when two items are
    deemed similar, union their sets.
    """

    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Union two sets by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def clusters(self) -> list[list[int]]:
        """Extract all clusters as lists of member indices."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            groups[self.find(i)].append(i)
        return list(groups.values())


# ---- Character trigrams ----


_BOUNDARY = "$$"


def char_trigrams(text: str) -> set[str]:
    """Generate character-level trigrams with boundary markers.

    Pads text with boundary markers to capture start/end patterns.
    Returns a set for efficient Jaccard computation.

    Args:
        text: Input string (should be pre-normalised to lowercase).

    Returns:
        Set of 3-character subsequences.

    Examples:
        >>> sorted(char_trigrams("plot"))
        ['$$p', '$pl', 'lot', 'ot$', 'plo', 't$$']
    """
    if not text:
        return set()
    padded = f"{_BOUNDARY}{text}{_BOUNDARY}"
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


# ---- Jaccard similarity ----


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index: |intersection| / |union|.

    Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def overlap_coefficient(set_a: set, set_b: set) -> float:
    """Overlap coefficient: |intersection| / min(|A|, |B|).

    Measures how much the smaller set is contained in the larger.
    Useful when strings share a core identity but differ in length
    (e.g. "Liberty Park" vs "Liberty Park Hoo Road").

    Returns 0.0 if either set is empty.
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return intersection / min(len(set_a), len(set_b))


def pairwise_jaccard(
    items: list[str],
) -> list[tuple[int, int, float]]:
    """Compute pairwise Jaccard similarity for a list of strings.

    Pre-computes character trigrams once per item, then compares
    all pairs. Suitable for small collections (100-1000 items).

    Args:
        items: List of strings to compare.

    Returns:
        List of (i, j, similarity) tuples for all pairs where
        similarity > 0. Pairs are ordered with i < j.
    """
    if len(items) < 2:
        return []

    # Pre-compute trigrams
    trigrams = [char_trigrams(item) for item in items]

    results = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = jaccard_similarity(trigrams[i], trigrams[j])
            if sim > 0.0:
                results.append((i, j, sim))

    return results
