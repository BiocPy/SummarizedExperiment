from typing import Sequence, Any
import numpy as np
import pandas as pd


def is_list_of_strings(obj) -> bool:
    """Returns `True` if obj is a list of strings.

    Args:
        obj: an object.
    """
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)


def get_indexes_from_names(
    _source: Sequence[Any], _target: Sequence[Any]
) -> np.ndarray:
    """Get the indexes in source where values in target are found.

    Args:
        _source (Sequence[Any]): the source index names.
        _target (Sequence[Any]): the target index names.

    Raises:
        ValueError: If passed index names do not exist in the samples.
        ValueError: If passed index names is not a collection.

    Returns:
        np.ndarray: Integers from 0 to n - 1 indicating that the index at
            these positions matches the corresponding target values. Missing
            values in the target are marked by -1.
    """
    try:
        source = pd.Index(_source)
        target = pd.Index(_target)
    except TypeError as exception:
        raise TypeError(f"{_target} is not a collection") from exception

    missing_names = target.difference(source)
    if not missing_names.empty:
        raise ValueError("invalid index name(s): " + ", ".join(missing_names))

    return source.get_indexer(target).tolist()
