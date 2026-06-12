"""Unit tests for graph_utils.py — DFS cycle detection."""

import pytest
from src.graph_utils import find_cycles


class TestFindCycles:
    """Test the DFS-based cycle finder for functional graphs."""

    def test_single_cycle_3_nodes(self):
        """Standard 3-node cycle: 1→2→3→1."""
        adj = {1: 2, 2: 3, 3: 1}
        cycles = find_cycles(adj)
        assert len(cycles) == 1
        assert cycles[0] == {1, 2, 3}

    def test_two_disjoint_cycles(self):
        """Two independent cycles: (1,2,3) and (4,5)."""
        adj = {1: 2, 2: 3, 3: 1, 4: 5, 5: 4}
        cycles = find_cycles(adj)
        assert len(cycles) == 2
        cycle_sets = [frozenset(c) for c in cycles]
        assert frozenset({1, 2, 3}) in cycle_sets
        assert frozenset({4, 5}) in cycle_sets

    def test_cycle_with_branches(self):
        """Structure: 3→4, 2→3, 1→2 (chain pointing into cycle 2→3→4→2)."""
        adj = {1: 2, 2: 3, 3: 4, 4: 2}
        cycles = find_cycles(adj)
        assert len(cycles) == 1
        assert cycles[0] == {2, 3, 4}

    def test_self_loop(self):
        """Agent votes for itself — should still be detected."""
        adj = {1: 1}
        cycles = find_cycles(adj)
        assert len(cycles) == 1
        assert cycles[0] == {1}

    def test_all_agents_in_one_cycle(self):
        """5 agents in a 5-cycle."""
        adj = {1: 2, 2: 3, 3: 4, 4: 5, 5: 1}
        cycles = find_cycles(adj)
        assert len(cycles) == 1
        assert len(cycles[0]) == 5
        assert cycles[0] == {1, 2, 3, 4, 5}

    def test_two_node_cycle(self):
        """Simple mutual voting: 1↔2."""
        adj = {1: 2, 2: 1}
        cycles = find_cycles(adj)
        assert len(cycles) == 1
        assert cycles[0] == {1, 2}

    def test_empty(self):
        """Empty graph."""
        assert find_cycles({}) == []

    def test_branch_pointing_into_cycle(self):
        """Agent 5→4→1 where 1-2-3 form a cycle."""
        adj = {1: 2, 2: 3, 3: 1, 4: 1, 5: 4}
        cycles = find_cycles(adj)
        assert len(cycles) == 1
        assert cycles[0] == {1, 2, 3}
