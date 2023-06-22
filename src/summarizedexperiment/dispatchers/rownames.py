from functools import singledispatch
from typing import Sequence

import pandas as pd
from biocframe import BiocFrame

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def get_rownames(obj) -> Sequence[str]:
    """Access row names from various objects.

    Args:
        obj (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Sequence[str]: column names.
    """
    if hasattr(obj, "rownames"):
        return obj.rownames

    raise NotImplementedError(f"rownames do not exist for class: {type(obj)}")


@get_rownames.register
def _(obj: pd.DataFrame) -> Sequence[str]:
    return obj.index.tolist()


@get_rownames.register
def _(obj: BiocFrame) -> Sequence[str]:
    return obj.rowNames


@singledispatch
def set_rownames(obj, names: Sequence[str]):
    """Set row names for various objects.

    Args:
        obj (any): supported object.
        names (Sequence[str]): new names.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Sequence[str]: column names.
    """
    raise NotImplementedError(f"Cannot set rownames for class: {type(obj)}")


@set_rownames.register
def _(obj: pd.DataFrame, names: Sequence[str]) -> Sequence[str]:
    obj.index = names
    return obj


@set_rownames.register
def _(obj: BiocFrame, names: Sequence[str]) -> Sequence[str]:
    obj.rowNames = names
    return obj
