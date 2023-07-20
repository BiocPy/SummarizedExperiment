from functools import singledispatch
import pandas as pd
from biocframe import BiocFrame


@singledispatch
def combine(
    left, right, ignore_names: bool = False, prefer_left: bool = False
) -> pd.DataFrame:
    """Combine various objects.

    Args:
        x (any): supported object.
    Raises:
        NotImplementedError: if type is not supported.
    Returns:
        pd.DataFrame: combined DataFrame.
    """
    raise NotImplementedError(f"cannot combine: {type(left)} and {type(right)}")


@combine.register
def _(
    left: pd.DataFrame, right, ignore_names: bool = False, prefer_left: bool = False
) -> pd.DataFrame:
    if ignore_names and prefer_left:
        raise ValueError("Somehow `useNames=True` and `useNames=False` simultaneously.")
    if not isinstance(right, pd.DataFrame):
        raise NotImplementedError(f"{type(right)} object are not supported")

    if ignore_names:
        return left.reset_index(drop=True).combine_first(right.reset_index(drop=True))
    elif prefer_left:
        return left.combine_first(right)
    else:
        return pd.concat([left, right])


@combine.register
def _(
    left: BiocFrame, right, ignore_names: bool = False, prefer_left: bool = False
) -> pd.DataFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")
