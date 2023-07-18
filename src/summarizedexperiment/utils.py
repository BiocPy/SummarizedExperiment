import numpy as np
import pandas as pd


def is_list_of_strings(obj) -> bool:
    """Returns `True` if obj is a list of strings.

    Args:
        obj: an object.
    """
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)


def get_indexes_from_names(source: pd.Index, target: pd.Index) -> np.ndarray:
    """Get the indexes in source where values in target are found.

    Args:
        source (pd.Index): the source index.
        target (pd.Index): the target index.

    Raises:
        ValueError: If passed index names do not exist in the samples.

    Returns:
        np.ndarray: Integers from 0 to n - 1 indicating that the index at
            these positions matches the corresponding target values. Missing
            values in the target are marked by -1.
    """
    missing_names = target.difference(source)
    if not missing_names.empty:
        raise ValueError(', '.join(missing_names) + "are invalid index names")

    return source.get_indexer(target)
