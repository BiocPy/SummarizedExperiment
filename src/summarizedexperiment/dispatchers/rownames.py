from functools import singledispatch
from typing import Any, Sequence

import pandas as pd
from biocframe import BiocFrame

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def get_rownames(x: Any) -> Sequence[str]:
    """Access row names from various objects.

    Args:
        x (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Sequence[str]: column names.
    """
    if hasattr(x, "rownames"):
        return x.rownames

    raise NotImplementedError(f"rownames do not exist for class: {type(x)}")


@get_rownames.register
def _(x: pd.DataFrame) -> Sequence[str]:
    return x.index.tolist()


@get_rownames.register
def _(x: BiocFrame) -> Sequence[str]:
    return x.rowNames


@singledispatch
def set_rownames(x: Any, names: Sequence[str]):
    """Set row names for various objects.

    Args:
        x (Any): supported object.
        names (Sequence[str]): new names.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Sequence[str]: column names.
    """
    raise NotImplementedError(f"Cannot set rownames for class: {type(x)}")


@set_rownames.register
def _(x: pd.DataFrame, names: Sequence[str]) -> Sequence[str]:
    x.index = names
    return x


@set_rownames.register
def _(x: BiocFrame, names: Sequence[str]) -> Sequence[str]:
    x.rowNames = names
    return x
