from typing import Union
from functools import singledispatch
import pandas as pd
from biocframe import BiocFrame


@singledispatch
def combine(left, right) -> Union[pd.DataFrame, BiocFrame]:
    """Combine various objects along the concatenation axis.

    Args:
        left (any): supported object.
        right (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Union[pd.DataFrame, BiocFrame]: combined DataFrame or BiocFrame.
    """
    raise NotImplementedError(
        f"cannot combine classes: {type(left)} and {type(right)}"
    )


@combine.register
def _(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(right, pd.DataFrame):
        raise NotImplementedError(f"{type(right)} object are not supported")
    return pd.concat([left, right])


@combine.register
def _(left: BiocFrame, right) -> BiocFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")


@singledispatch
def combine_other(left, right) -> Union[pd.DataFrame, BiocFrame]:
    """Combine various objects along the non-concatenation axis.

    Args:
        left (any): supported object.
        right (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        Union[pd.DataFrame, BiocFrame]: combined DataFrame or BiocFrame.
    """
    raise NotImplementedError(
        f"cannot combine classes: {type(left)} and {type(right)}"
    )

@combine_other.register
def _(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(right, pd.DataFrame):
        raise NotImplementedError(f"{type(right)} object are not supported")
    return left.combine_first(right)

@combine_other.register
def _(left: BiocFrame, right) -> BiocFrame:
    raise NotImplementedError("BiocFrame objects are currently not supported.")