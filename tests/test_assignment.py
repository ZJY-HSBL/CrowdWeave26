"""CRA coverage and safety invariants independent of Ultralytics."""
import pytest
from crowdweave26.assignment import repair_coverage


def test_relocates_donor_to_free_alternative():
    owners, stats = repair_coverage([0, -1], [[0, 1], [0]])
    assert owners == [1, 0]
    assert stats["recovered"] == 1
    assert stats["orphans_after"] == 0


def test_preserves_donor_with_multiple_positives():
    owners, stats = repair_coverage([0, 0, -1], [[0, 1], [0]])
    assert owners == [1, 0, -1]
    assert stats["recovered"] == 1


def test_chain_of_two_relocations():
    owners, stats = repair_coverage([0, 1, -1], [[0, 1], [1, 2], [0]])
    assert owners == [2, 0, 1]
    assert stats["recovered"] == 1


def test_infeasible_collision_does_not_orphan_donor():
    owners, stats = repair_coverage([0], [[0], [0]])
    assert owners == [0]
    assert stats["orphans_before"] == 1 and stats["orphans_after"] == 1


def test_no_orphan_is_bit_identical():
    owners, stats = repair_coverage([0, 1, 0], [[0, 2], [1]])
    assert owners == [0, 1, 0] and stats["changed_anchors"] == 0


def test_empty_and_unassignable_gt():
    assert repair_coverage([], [[], []])[0] == []
    assert repair_coverage([-1, -1], [[], []])[0] == [-1, -1]


def test_input_validation():
    with pytest.raises(ValueError):
        repair_coverage([2], [[0]])
    with pytest.raises(ValueError):
        repair_coverage([-1], [[0, 0]])
    with pytest.raises(ValueError):
        repair_coverage([-1], [[1]])


def test_original_owners_unchanged():
    owners = [0, -1]
    candidates = [[0, 1], [0]]
    repair_coverage(owners, candidates)
    assert owners == [0, -1]


def test_randomized_coverage_and_candidate_safety():
    import random
    rng = random.Random(20261015)
    for _ in range(150):
        n_gt = rng.randint(1, 18)
        n_anchor = rng.randint(1, 36)
        original = [rng.randrange(-1, n_gt) for _ in range(n_anchor)]
        candidates = [rng.sample(range(n_anchor), rng.randint(0, min(n_anchor, 6)))
                      for _ in range(n_gt)]
        updated, stats = repair_coverage(original, candidates)
        before = {g for g in original if g >= 0}
        after = {g for g in updated if g >= 0}
        assert before.issubset(after)
        assert len(after) >= len(before)
        assert len(updated) == len(original)
        for anchor, (old, new) in enumerate(zip(original, updated)):
            if old != new:
                assert new >= 0 and anchor in candidates[new]
        assert stats["recovered"] >= 0


def test_long_chain_no_recursion_overflow():
    n = 1100
    owners = list(range(n-1)) + [-1]
    candidates = [[i, i+1] for i in range(n-1)] + [[0]]
    updated, stats = repair_coverage(owners, candidates)
    assert stats["recovered"] == 1
    assert len(set(updated)) == n
