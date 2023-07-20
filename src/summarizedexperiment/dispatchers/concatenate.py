from functools import singledispatch
import pandas as pd
from biocframe import BiocFrame


@singledispatch
def concatenate(left, right) -> pd.DataFrame:
    """Concatenate various objects together.

    Args:
        x (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        pd.DataFrame: concatenated DataFrame.
    """
    raise NotImplementedError(
        f"cannot concatenate classes: {type(left)} and {type(right)}"
    )


@concatenate.register
def _(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([left, right])


@concatenate.register
def _(left: BiocFrame, right) -> pd.DataFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")


@concatenate.register
def _(left, right: BiocFrame) -> pd.DataFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")
