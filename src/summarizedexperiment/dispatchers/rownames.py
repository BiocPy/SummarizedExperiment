from functools import singledispatch
from typing import Any, List, Sequence

from biocframe import BiocFrame
from pandas import DataFrame

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def get_rownames(x) -> List[str]:
    """Access row names from various representations.

    Args:
        x: Any object.

        ``x`` may be a :py:class:`~pandas.DataFrame`.

        Alternatively, ``x`` may be a :py:class:`~biocframe.BiocFrame.BiocFrame` object.

        Alternatively, ``x`` may also contain a property or attribute ``row_names`` for
        custom representations.

    Raises:
        NotImplementedError: If ``x`` is not a supported type.

    Returns:
        List[str]: List of row names.
    """
    if hasattr(x, "row_names"):
        return x.row_names

    raise NotImplementedError(f"`row_names` do not exist for class: '{type(x)}'.")


@get_rownames.register
def _(x: DataFrame) -> List[str]:
    return x.index.tolist()


@get_rownames.register
def _(x: BiocFrame) -> List[str]:
    return x.row_names


@singledispatch
def set_rownames(x: Any, names: Sequence[str]):
    """Set row names for various representations.

    Args:
        x (Any): supported object.

        ``x`` may be a :py:class:`~pandas.DataFrame`.

        Alternatively, ``x`` may be a :py:class:`~biocframe.BiocFrame.BiocFrame` object.

        Alternatively, ``x`` may also contain a property or attribute ``row_names`` for
        custom representations.

        names (Sequence[str]): New names.

    Raises:
        NotImplementedError: If ``x`` is not a supported type.

    Returns:
        An object with the same type as ``x``.
    """
    raise NotImplementedError(f"Cannot set row_names for class: {type(x)}")


@set_rownames.register
def _(x: DataFrame, names: Sequence[str]) -> Sequence[str]:
    x.index = names
    return x


@set_rownames.register
def _(x: BiocFrame, names: Sequence[str]) -> Sequence[str]:
    x.row_names = names
    return x
