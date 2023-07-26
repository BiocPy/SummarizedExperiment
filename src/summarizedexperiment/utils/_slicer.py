from typing import Any, Sequence

import numpy as np
import pandas as pd

__author__ = "keviny2, jkanche"
__copyright__ = "Genentech"
__license__ = "MIT"


def get_indexes_from_names(
    source: Sequence[Any], target: Sequence[Any]
) -> Sequence[int]:
    """Return the index of the first occurrence of each value in `source`.

    Args:
        source (Sequence[Any]): the list in which to search for the values.
        target (Sequence[Any]): the list of values to find indices for.

    Raises:
        ValueError: if any value in `target` is not found in `source`.

    Returns:
        Sequence[int]: the indexes of the first occurrence of each value in
            `target` within `source`.
    """
    missing_names = set(target).difference(source)
    if len(missing_names) > 0:
        raise ValueError("invalid index names(s): " + ", ".join(missing_names))

    value_to_index = {}
    for index, value in enumerate(source):
        if value not in value_to_index:
            value_to_index[value] = index
    
    return [value_to_index[value] for value in target]


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
