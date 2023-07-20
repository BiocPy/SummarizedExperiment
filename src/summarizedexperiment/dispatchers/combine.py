from functools import singledispatch
import pandas as pd
from biocframe import BiocFrame


@singledispatch
def combine(left, right) -> pd.DataFrame:
    """Combine various objects together.

    Args:
        x (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        pd.DataFrame: combined DataFrame.
    """
    raise NotImplementedError(
        f"cannot combine classes: {type(left)} and {type(right)}"
    )


@combine.register
def _(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([left, right])


@combine.register
def _(left: BiocFrame, right) -> pd.DataFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")


@combine.register
def _(left, right: BiocFrame) -> pd.DataFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")
