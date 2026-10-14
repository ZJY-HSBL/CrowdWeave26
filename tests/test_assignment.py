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
