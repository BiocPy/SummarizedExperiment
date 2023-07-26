from typing import Any, Sequence

import pandas as pd

__author__ = "keviny2, jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def get_indexes_from_names(
    source: Sequence[Any], target: Sequence[Any], firstMatch: bool = True
) -> Sequence[int]:
    """Return the index of the first occurrence of each value in `source`.

    Args:
        source (Sequence[Any]): list in which to search for the values.
        target (Sequence[Any]): list of values to find indices for.
        firstMatch (bool): only return first matches? Defaults to True.

    Raises:
        ValueError: if any value in `target` is not found in `source`.
        ValueError: if target contains duplicates.

    Returns:
        Sequence[int]: the indexes of the first occurrence of each value in
            `target` within `source`.
    """

    if isinstance(source, pd.Index):
        source = source.tolist()

    if isinstance(target, pd.Index):
        target = target.tolist()

    set_target = set(target)

    missing_names = set_target.difference(source)
    if len(missing_names) > 0:
        raise ValueError(f"invalid index names(s): {', '.join(missing_names)}")

    if len(set_target) != len(target):
        raise ValueError("target contains duplicate values.")

    if firstMatch is True:
        return [source.index(x) for x in target]

    match_indices = []
    for v in target:
        match_indices.extend([i for i, x in enumerate(source) if x == v])

    return list(set(match_indices))


def get_indexes_from_bools(x: Sequence[bool], match: bool = True) -> Sequence[int]:
    """Get indices where values in `x` are equal to `match`.

    if match is `True`, returns indices in `x` whose value is `True`.

    Args:
        x (Sequence[bool]): boolean vector.
        match (bool, optional): value to find indices for. Defaults to True.

    Returns:
        Sequence[int]: list of indices with matches
    """
    return [i for i in range(len(x)) if x[i] == match]
