from typing import Union
from functools import singledispatch
import pandas as pd
from biocframe import BiocFrame


@singledispatch
def combine(left, right) -> pd.DataFrame:
    """Combine various objects.

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


@combine.register
def _(left: pd.DataFrame, right) -> pd.DataFrame:
    if not isinstance(right, pd.DataFrame):
        raise NotImplementedError(f"{type(right)} object are not supported")
    return pd.concat([left, right])


@combine.register
def _(left: BiocFrame, right) -> pd.DataFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")


@singledispatch
def combine_prefer_left(left, right) -> Union[pd.DataFrame, BiocFrame]:
    """Combine various objects, prefering values from the left dataframe when
    there are overlapping columns.

    Args:
        left (any): supported object.
        right (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Union[pd.DataFrame, BiocFrame]: combined DataFrame or BiocFrame.
    """
    raise NotImplementedError(f"cannot combine classes: {type(left)} and {type(right)}")


@combine_prefer_left.register
def _(left: pd.DataFrame, right) -> pd.DataFrame:
    if not isinstance(right, pd.DataFrame):
        raise NotImplementedError(f"{type(right)} object are not supported")
    return left.combine_first(right)


@combine_prefer_left.register
def _(left: BiocFrame, right) -> BiocFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")


@singledispatch
def combine_ignore_names(left, right) -> Union[pd.DataFrame, BiocFrame]:
    """Combine various objects, ignoring index names.

    Args:
        left (any): supported object.
        right (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Union[pd.DataFrame, BiocFrame]: combined DataFrame or BiocFrame.
    """
    raise NotImplementedError(f"cannot combine classes: {type(left)} and {type(right)}")


@combine_ignore_names.register
def _(left: pd.DataFrame, right) -> pd.DataFrame:
    if not isinstance(right, pd.DataFrame):
        raise NotImplementedError(f"{type(right)} object are not supported")
    return left.reset_index(drop=False).combine_first(right.reset_index(drop=False))


@combine_ignore_names.register
def _(left: BiocFrame, right) -> BiocFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")
