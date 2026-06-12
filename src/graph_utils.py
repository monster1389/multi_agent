"""Graph utilities for deadlock resolution — DFS cycle detection.

In the endgame phase, each agent votes for exactly one other agent,
forming a functional graph (each node out-degree = 1, a "基环树").
By graph theory, every such finite directed graph must contain at least one cycle.
This module finds all cycles so we can break deadlocks by eliminating
the lowest-Γ agent within the cycle.
"""

from collections import defaultdict


# Three-color DFS states
_WHITE = 0   # unvisited
_GRAY = 1    # in current recursion stack
_BLACK = 2   # fully processed


def find_cycles(adjacency: dict[int, int]) -> list[set[int]]:
    """Find all cycles in a functional graph (each node has out-degree 1).

    Uses DFS three-color marking to detect back edges, then traces each cycle.

    Args:
        adjacency: Mapping agent_id → voted_for_agent_id.
                   Every key must have a value; out-degree is strictly 1.

    Returns:
        List of sets, each set containing the agent_ids in one cycle.
        Cycles that share nodes are merged. Returns empty list if no cycles.

    Example:
        >>> find_cycles({1:2, 2:3, 3:1, 4:5, 5:4})
        [{1, 2, 3}, {4, 5}]
        >>> find_cycles({1:2, 2:1, 3:2})
        [{1, 2}]
        >>> find_cycles({1:2, 2:3, 3:4, 4:1, 5:1})
        [{1, 2, 3, 4}]
    """
    color: dict[int, int] = defaultdict(lambda: _WHITE)
    parent: dict[int, int | None] = {}
    cycles: list[set[int]] = []

    def dfs(node: int) -> None:
        color[node] = _GRAY
        neighbor = adjacency.get(node)
        if neighbor is None:
            color[node] = _BLACK
            return

        if color[neighbor] == _GRAY:
            # Back edge found — trace the cycle
            cycle: set[int] = set()
            current = node
            while current != neighbor:
                cycle.add(current)
                current = parent.get(current, neighbor)
                # Safety check: if we lose the parent chain, break
                if current is None:
                    break
            cycle.add(neighbor)
            cycles.append(cycle)

        elif color[neighbor] == _WHITE:
            parent[neighbor] = node
            dfs(neighbor)

        color[node] = _BLACK

    # Start DFS from every unvisited node
    for node in list(adjacency.keys()):
        if color[node] == _WHITE:
            dfs(node)

    # Merge overlapping cycles (nodes that appear in multiple cycles
    # are part of the same connected cycle component)
    merged = _merge_overlapping_cycles(cycles)
    return merged


def _merge_overlapping_cycles(cycles: list[set[int]]) -> list[set[int]]:
    """Merge cycles that share at least one node (union-find style)."""
    if not cycles:
        return []

    # Simple iterative merging
    result: list[set[int]] = []
    for cycle in cycles:
        merged_with = None
        for i, existing in enumerate(result):
            if cycle & existing:  # non-empty intersection
                merged_with = i
                break

        if merged_with is not None:
            result[merged_with] |= cycle
            # Re-check for new overlaps after merging
            result = _merge_overlapping_cycles(result)
            return result
        else:
            result.append(cycle)

    return result
