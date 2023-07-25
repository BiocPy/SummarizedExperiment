from typing import Sequence

from ..types import BiocOrPandasFrame

__author__ = "keviny2, jkanche"
__copyright__ = "keviny2"
__license__ = "MIT"


def _validate_single_frame(df: BiocOrPandasFrame) -> bool:
    """Validate there are no None or duplicated names.

    Args:
        df (BiocOrPandasFrame): pandas or Bioc DataFrame to validate.

    Returns:
        bool: `True` if df does not have any None or duplicated names.
            `False` otherwise.
    """
    any_null = any(name is None for name in df.index)
    any_duplicates = len(df.index) != len(set(df.index))
    return (not any_null) and (not any_duplicates)


def validate_names(x: Sequence[BiocOrPandasFrame]):
    """Validate names across experiments.

    Args:
        x (Sequence[BiocOrPandasFrame]): BiocOrPandasFrame objects to validate.

    Raises:
        ValueError: if there are null or duplicated names.
    """

    is_valid_names = all([_validate_single_frame(se) for se in x])
    if not is_valid_names:
        raise ValueError("at least one input has null or duplicated row names")


def validate_shapes(x: Sequence[BiocOrPandasFrame]):
    """Validate shapes across experiments.

    Args:
        x (Sequence[BiocOrPandasFrame]): BiocOrPandasFrame objects to validate.

    Raises:
        ValueError: if all objects do not have the same shape of interest:
            - number of rows for cbind() and combineCols()
            - number of columns for rbind() and combineRows()
    """
    all_shapes = [f.shape[0] for f in x]
    is_all_same_shape = all_shapes.count(all_shapes[0]) == len(all_shapes)
    if not is_all_same_shape:
        raise ValueError("not all objects have the same shape")
