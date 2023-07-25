from functools import singledispatch
from typing import Any, Sequence

import pandas as pd
from biocframe import BiocFrame

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def get_colnames(x: Any) -> Sequence[str]:
    """Access column names from various objects.

    Args:
        x (Any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Sequence[str]: column names.
    """
    if hasattr(x, "colnames"):
        return x.colnames

    raise NotImplementedError(f"colnames do not exist for class: {type(x)}")


@get_colnames.register
def _(x: pd.DataFrame) -> Sequence[str]:
    return x.index.tolist()


@get_colnames.register
def _(x: BiocFrame) -> Sequence[str]:
    return x.rowNames


@singledispatch
def set_colnames(x: Any, names: Sequence[str]):
    """Set column names for various objects.

    Args:
        x (any): supported object.
        names (Sequence[str]): new names.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Sequence[str]: column names.
    """
    raise NotImplementedError(f"Cannot set colnames for class: {type(x)}")


@set_colnames.register
def _(x: pd.DataFrame, names: Sequence[str]) -> Sequence[str]:
    x.index = names
    return x


@set_colnames.register
def _(x: BiocFrame, names: Sequence[str]) -> Sequence[str]:
    x.rowNames = names
    return x
