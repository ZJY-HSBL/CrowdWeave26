"""Sparse collision repair for dense pedestrian positive assignments.

A deterministic augmenting-path repair over existing TAL candidates. No new
positive candidates are fabricated, and no covered GT loses its final anchor.
The original assignment is returned unchanged if no feasible repair exists.
"""
from __future__ import annotations
from collections import deque


def repair_coverage(owners: list[int], candidates: list[list[int]]) -> tuple[list[int], dict]:
    """Repair zero-positive GTs using candidate-to-anchor alternating paths.

    owners[a] is a GT index or -1 for an unassigned anchor. candidates[g] is a
    quality-ranked list of anchor indices from the *original* TAL top-k mask.
    Existing assignments are retained unless an alternating path can improve
    GT coverage without reducing coverage of any previously represented GT.
    """
    n_gt, n_anchor = len(candidates), len(owners)
    current = list(owners)
    if any(not isinstance(g, int) or g < -1 or g >= n_gt for g in current):
        raise ValueError("owner index outside [-1, number of GTs)")
    for group in candidates:
        if len(set(group)) != len(group) or any(
            not isinstance(a, int) or a < 0 or a >= n_anchor for a in group
        ):
            raise ValueError("invalid or duplicate candidate anchor")
    counts = [0] * n_gt
    for g in current:
        if g >= 0:
            counts[g] += 1
    initial = list(current)
    orphans = [g for g in range(n_gt) if counts[g] == 0 and candidates[g]]
    # Fewest alternatives first, then annotation index: deterministic ordering.
    for start in sorted(orphans, key=lambda g: (len(candidates[g]), g)):
        if counts[start] != 0:
            continue
        queue = deque([start])
        visited_gts = {start}
        visited_anchors = set()
        parent = {}  # displaced_gt -> (previous_gt, anchor_owned_by_displaced_gt)
        terminal = None
        while queue and terminal is None:
            gt = queue.popleft()
            for anchor in candidates[gt]:
                if anchor in visited_anchors:
                    continue
                visited_anchors.add(anchor)
                donor = current[anchor]
                if donor == -1 or counts[donor] > 1:
                    terminal = (gt, anchor)
                    break
                if donor not in visited_gts:
                    visited_gts.add(donor)
                    parent[donor] = (gt, anchor)
                    queue.append(donor)
        if terminal is None:
            continue
        gt, anchor = terminal
        while True:
            donor = current[anchor]
            if donor != gt:
                if donor >= 0:
                    counts[donor] -= 1
                current[anchor] = gt
                counts[gt] += 1
            if gt == start:
                break
            gt, anchor = parent[gt]
    after = sum(counts[g] == 0 for g in orphans)
    return current, {
        "orphans_before": len(orphans),
        "orphans_after": after,
        "recovered": len(orphans) - after,
        "changed_anchors": sum(a != b for a, b in zip(initial, current)),
    }
