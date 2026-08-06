from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.manipulation.leap_inhand.deploy_contract import (
    NUM_JOINTS,
    invert_permutation,
    reorder_source_to_destination,
    validate_permutation,
)


def test_identity_permutation() -> None:
    identity = np.arange(NUM_JOINTS)
    values = np.arange(NUM_JOINTS) * 10

    np.testing.assert_array_equal(validate_permutation(identity), identity)
    np.testing.assert_array_equal(reorder_source_to_destination(values, identity), values)
    np.testing.assert_array_equal(invert_permutation(identity), identity)


def test_nontrivial_permutation_inverse_and_exact_round_trip() -> None:
    mapping = np.array([2, 0, 3, 1])
    source = np.array([10, 20, 30, 40])

    destination = reorder_source_to_destination(source, mapping)
    inverse = invert_permutation(mapping, size=4)
    restored = reorder_source_to_destination(destination, inverse)

    np.testing.assert_array_equal(destination, [20, 40, 10, 30])
    np.testing.assert_array_equal(inverse, [1, 3, 0, 2])
    np.testing.assert_array_equal(restored, source)


def test_mapping_reorder_supports_batch() -> None:
    mapping = np.array([2, 0, 3, 1])
    source = np.array([[10, 20, 30, 40], [11, 21, 31, 41]])

    destination = reorder_source_to_destination(source, mapping)

    np.testing.assert_array_equal(destination[0], [20, 40, 10, 30])
    np.testing.assert_array_equal(destination[1], [21, 41, 11, 31])


@pytest.mark.parametrize(
    "mapping",
    [
        np.array([0, 1, 1, 3]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2, 4]),
        np.array([-1, 0, 1, 2]),
    ],
)
def test_invalid_permutations_are_rejected(mapping) -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_permutation(mapping, size=4)


def test_noninteger_permutation_is_rejected() -> None:
    with pytest.raises(TypeError, match="integer"):
        validate_permutation(np.array([0.0, 1.0, 2.0, 3.0]), size=4)
