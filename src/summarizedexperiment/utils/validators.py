from typing import Sequence

from ..types import BiocOrPandasFrame

__author__ = "keviny2, jkanche"
__copyright__ = "keviny2"
__license__ = "MIT"


def _validate_single_frame(df: BiocOrPandasFrame) -> bool:
    """Validate there are no None or duplicated names.

    Args:
        df (BiocOrPandasFrame): Object to validate.
            ``df`` may be a :py;class:`~biocframe.BiocFrame.BiocFrame` object.
            Alternatively, ``df`` may be a :py:class:`~pandas.DataFrame`.

    Returns:
        bool: `True` if ``df`` does not have any `None` or duplicated names.
        `False` otherwise.
    """
    any_null = any(name is None for name in df.index)
    any_duplicates = len(df.index) != len(set(df.index))
    return (not any_null) and (not any_duplicates)


def validate_names(x: Sequence[BiocOrPandasFrame]) -> bool:
    """Validate names across experiments.

    Args:
        x (Sequence[BiocOrPandasFrame]): Objects to validate.
            ``x`` may be a :py;class:`~biocframe.BiocFrame.BiocFrame` object.
            Alternatively, ``x`` may be a :py:class:`~pandas.DataFrame`.

    Raises:
        ValueError: If there are :py:obj:`~pandas.nan`, None or duplicated names.

    Returns:
        bool: `True` if ``x`` does not have any `None` or duplicated names.
        `False` otherwise.
    """

    is_valid_names = all([_validate_single_frame(se) for se in x])
    if not is_valid_names:
        raise ValueError("At least one input has null or duplicated row names.")

    return is_valid_names


def validate_shapes(x: Sequence[BiocOrPandasFrame]):
    """Validate shapes across experiments.

    Args:
        x (Sequence[BiocOrPandasFrame]): Objects to validate.
            ``x`` may be a :py;class:`~biocframe.BiocFrame.BiocFrame` object.
            Alternatively, ``x`` may be a :py:class:`~pandas.DataFrame`.

    Raises:
        ValueError: if all objects do not have the same shape.
            - number of rows for
                :py:meth:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.combine_cols`.
            - number of columns for
                :py:meth:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.combine_rows`.
    """
    all_shapes = [f.shape[0] for f in x]
    is_all_same_shape = all_shapes.count(all_shapes[0]) == len(all_shapes)
    if not is_all_same_shape:
        raise ValueError("Not all objects have the same shape.")
