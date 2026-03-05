"""Tests for character-level similarity utilities."""

import pytest

from classification.similarity import (
    UnionFind,
    char_trigrams,
    jaccard_similarity,
    overlap_coefficient,
    pairwise_jaccard,
)


# ---- char_trigrams ----


class TestCharTrigrams:
    def test_basic_string(self):
        result = char_trigrams("plot")
        assert "$$p" in result
        assert "$pl" in result
        assert "plo" in result
        assert "lot" in result
        assert "ot$" in result
        assert "t$$" in result

    def test_empty_string(self):
        assert char_trigrams("") == set()

    def test_single_char(self):
        result = char_trigrams("a")
        # "$$a$$" -> "$$a", "$a$", "a$$"
        assert result == {"$$a", "$a$", "a$$"}

    def test_two_chars(self):
        result = char_trigrams("ab")
        # "$$ab$$" -> "$$a", "$ab", "ab$", "b$$"
        assert result == {"$$a", "$ab", "ab$", "b$$"}

    def test_boundary_markers_present(self):
        result = char_trigrams("hello")
        # Should have start boundary
        assert any(t.startswith("$$") for t in result)
        # Should have end boundary
        assert any(t.endswith("$$") for t in result)

    def test_returns_set(self):
        result = char_trigrams("test")
        assert isinstance(result, set)

    def test_spaces_preserved(self):
        result = char_trigrams("a b")
        assert "a b" in result

    def test_similar_strings_share_trigrams(self):
        a = char_trigrams("lenham road")
        b = char_trigrams("lenham rd")
        shared = a & b
        assert len(shared) > 0


# ---- jaccard_similarity ----


class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        # {a, b, c} ∩ {b, c, d} = {b, c} -> 2
        # {a, b, c} ∪ {b, c, d} = {a, b, c, d} -> 4
        assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_both_empty(self):
        assert jaccard_similarity(set(), set()) == 0.0

    def test_one_empty(self):
        assert jaccard_similarity({"a"}, set()) == 0.0

    def test_subset(self):
        # {a, b} ∩ {a, b, c} = {a, b} -> 2
        # {a, b} ∪ {a, b, c} = {a, b, c} -> 3
        result = jaccard_similarity({"a", "b"}, {"a", "b", "c"})
        assert abs(result - 2 / 3) < 1e-9


# ---- overlap_coefficient ----


class TestOverlapCoefficient:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert overlap_coefficient(s, s) == 1.0

    def test_subset(self):
        """Smaller set fully contained in larger -> 1.0."""
        assert overlap_coefficient({"a", "b"}, {"a", "b", "c", "d"}) == 1.0

    def test_disjoint_sets(self):
        assert overlap_coefficient({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        # {a, b, c} ∩ {b, c, d} = {b, c} -> 2
        # min(3, 3) = 3
        result = overlap_coefficient({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(result - 2 / 3) < 1e-9

    def test_empty_a(self):
        assert overlap_coefficient(set(), {"a"}) == 0.0

    def test_empty_b(self):
        assert overlap_coefficient({"a"}, set()) == 0.0

    def test_both_empty(self):
        assert overlap_coefficient(set(), set()) == 0.0

    def test_asymmetric_sizes(self):
        """Overlap is relative to the smaller set."""
        small = {"a", "b"}
        large = {"a", "b", "c", "d", "e"}
        assert overlap_coefficient(small, large) == 1.0
        assert overlap_coefficient(large, small) == 1.0  # symmetric


# ---- pairwise_jaccard ----


class TestPairwiseJaccard:
    def test_two_identical(self):
        result = pairwise_jaccard(["hello", "hello"])
        assert len(result) == 1
        assert result[0] == (0, 1, 1.0)

    def test_two_different(self):
        result = pairwise_jaccard(["aaa", "zzz"])
        # Very different strings should have low similarity
        sims = [r[2] for r in result]
        for s in sims:
            assert s < 0.3

    def test_three_items(self):
        result = pairwise_jaccard(["abc", "abd", "xyz"])
        # Should have 3 pairs: (0,1), (0,2), (1,2)
        pairs = {(r[0], r[1]) for r in result}
        # At least (0,1) should exist (abc and abd are similar)
        assert (0, 1) in pairs

    def test_single_item(self):
        assert pairwise_jaccard(["hello"]) == []

    def test_empty_list(self):
        assert pairwise_jaccard([]) == []

    def test_ordering(self):
        result = pairwise_jaccard(["a", "b", "c"])
        for i, j, _ in result:
            assert i < j


# ---- UnionFind ----


class TestUnionFind:
    def test_initial_state(self):
        uf = UnionFind(5)
        # Each element is its own root
        for i in range(5):
            assert uf.find(i) == i

    def test_union_and_find(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_transitive(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_separate_groups(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) != uf.find(2)

    def test_clusters(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        clusters = uf.clusters()
        assert len(clusters) == 3  # {0,1}, {2,3}, {4}
        cluster_sets = [set(c) for c in clusters]
        assert {0, 1} in cluster_sets
        assert {2, 3} in cluster_sets
        assert {4} in cluster_sets

    def test_single_element(self):
        uf = UnionFind(1)
        assert uf.clusters() == [[0]]

    def test_all_merged(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        clusters = uf.clusters()
        assert len(clusters) == 1
        assert set(clusters[0]) == {0, 1, 2, 3}

    def test_idempotent_union(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        uf.union(0, 1)  # Duplicate union
        clusters = uf.clusters()
        assert len(clusters) == 2
