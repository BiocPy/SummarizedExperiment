from typing import Any, List, Sequence, Union

from pandas import Index

__author__ = "keviny2, jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def get_indexes_from_names(
    source: Union[Index, Sequence[Any]],
    target: Union[Index, Sequence[Any]],
    first_match: bool = True,
) -> List[Union[int, List[int]]]:
    """Return the index of the first occurrence of each value in ``source``.

    Elements in ``source`` and ``target`` must be comparable.

    Args:
        source (Union[Index, Sequence[Any]]): List in which to search for the values.
            Alternatively, ``source`` may be a :py:class:`~pandas.Index` object.
        target (Union[Index, Sequence[Any]]): List of values to find indices for.
            Alternatively, ``target`` may be a :py:class:`~pandas.Index` object.
        first_match (bool): Whether to only return first matches. Defaults to True.

    Raises:
        ValueError: If any value in ``target`` is not found in ``source``.
        ValueError: If ``target`` contains duplicates.

    Returns:
        List[Union[int, List[int]]]: A list with the same length as target,
        with each element the positional indexes of the occurrence of the corresponding value from
        ``target`` within ``source``.

        If ``first_match`` is False, the list might contain be a list of indices.
    """

    if isinstance(source, Index):
        source = source.tolist()

    if isinstance(target, Index):
        target = target.tolist()

    set_target = set(target)

    missing_names = set_target.difference(source)
    if len(missing_names) > 0:
        raise ValueError(
            f"Invalid index names(s) in `source`: {', '.join(missing_names)}."
        )

    if len(set_target) != len(target):
        raise ValueError("`target` contains duplicate values.")

    if first_match is True:
        return [source.index(x) for x in target]

    # TODO: use the bisect module to do better.
    match_indices = []
    for v in target:
        match_indices.extend([i for i, x in enumerate(source) if x == v])

    return list(set(match_indices))


def get_indexes_from_bools(x: Sequence[bool], match: bool = True) -> List[int]:
    """Get positional indices where values in ``x`` are equal to ``match``.

    If ``match`` is `True`, returns indices in ``x`` whose value is `True`.

    Args:
        x (Sequence[bool]): Boolean vector.
        match (bool, optional): Value to find indices for. Defaults to ``True``.

    Returns:
        List[int]: List of indices with matches.
    """
    return [i for i in range(len(x)) if x[i] == match]
