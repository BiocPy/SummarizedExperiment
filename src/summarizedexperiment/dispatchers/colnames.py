from functools import singledispatch
from typing import List, Sequence

from biocframe import BiocFrame
from pandas import DataFrame

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def get_colnames(x) -> List[str]:
    """Access column names from various representations.

    Args:
        x: Any object.

            ``x`` may be a :py:class:`~pandas.DataFrame`.

            Alternatively, ``x`` may be a :py:class:`~biocframe.BiocFrame.BiocFrame` object.

            Alternatively, ``x`` may also contain a property or attribute ``colnames`` for
            custom representations.

    Raises:
        NotImplementedError: If ``x`` is not a supported type.

    Returns:
        List[str]: List of column names.
    """
    if hasattr(x, "colnames"):
        return x.colnames

    raise NotImplementedError(f"`colnames` is not supported for class: '{type(x)}'.")


@get_colnames.register
def _(x: DataFrame) -> List[str]:
    return x.index.tolist()


@get_colnames.register
def _(x: BiocFrame) -> List[str]:
    return x.row_names


@singledispatch
def set_colnames(x, names: List[str]):
    """Set column names for various representations.

    Args:
        x: Any object.

            ``x`` may be a :py:class:`~pandas.DataFrame`.

            Alternatively, ``x`` may be a :py:class:`biocframe.BiocFrame.BiocFrame` object.

            Alternatively, ``x`` may also contain a property or attribute ``colnames`` for
            custom representations.

        names (Sequence[str]): New names.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        An object with the same type as ``x``.
    """
    raise NotImplementedError(
        f"`set_colnames` is not supported for class: '{type(x)}'."
    )


@set_colnames.register
def _(x: DataFrame, names: Sequence[str]) -> DataFrame:
    x.index = names
    return x


@set_colnames.register
def _(x: BiocFrame, names: Sequence[str]) -> BiocFrame:
    x.row_names = names
    return x
