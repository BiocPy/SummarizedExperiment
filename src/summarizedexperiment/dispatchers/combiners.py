from multipledispatch import dispatch
import pandas as pd
from biocframe import BiocFrame

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def validate_inputs(ignore_names: bool, prefer_left: bool):
    """Ensure boolean values are valid.

    Args:
        ignore_names (bool): if `True`, index names will be ignored when combining.
        prefer_left (bool): if `True`, only null values in the left dataframe
            will be populated by values from the right dataframe.
    """
    if ignore_names and prefer_left:
        raise ValueError("`ignore_names` and `prefer_left` cannot both be `True`")


def combine_dataframes(
    left: pd.DataFrame, right: pd.DataFrame, ignore_names: bool, prefer_left: bool
) -> pd.DataFrame:
    """Combine dataframes with the appropriate method.

    Args:
        left (pd.DataFrame): a dataframe.
        right (pd.DataFrame): a dataframe.
        ignore_names (bool): if `True`, index names will be ignored when combining.
        prefer_left (bool): if `True`, only null values in the left dataframe
            will be populated by values from the right dataframe.
    Returns:
        pd.DataFrame: the combined dataframe.
    """
    validate_inputs(ignore_names, prefer_left)
    if ignore_names:
        return left.reset_index(drop=True).combine_first(right.reset_index(drop=True))
    if prefer_left:
        return left.combine_first(right)
    return pd.concat([left, right])


@dispatch(pd.DataFrame, pd.DataFrame)
def combine(
    left: pd.DataFrame,
    right: pd.DataFrame,
    ignore_names: bool = False,
    prefer_left: bool = False,
) -> pd.DataFrame:
    return combine_dataframes(
        left, right, ignore_names=ignore_names, prefer_left=prefer_left
    )


@dispatch(BiocFrame, BiocFrame)
def combine(
    left: BiocFrame,
    right: BiocFrame,
    ignore_names: bool = False,
    prefer_left: bool = False,
) -> pd.DataFrame:
    return combine_dataframes(
        left.to_pandas(),
        right.to_pandas(),
        ignore_names=ignore_names,
        prefer_left=prefer_left,
    )


@dispatch(pd.DataFrame, BiocFrame)
def combine(
    left: pd.DataFrame,
    right: BiocFrame,
    ignore_names: bool = False,
    prefer_left: bool = False,
) -> pd.DataFrame:
    return combine_dataframes(
        left, right.to_pandas(), ignore_names=ignore_names, prefer_left=prefer_left
    )


@dispatch(BiocFrame, pd.DataFrame)
def combine(
    left: BiocFrame,
    right: BiocFrame,
    ignore_names: bool = False,
    prefer_left: bool = False,
) -> pd.DataFrame:
    return combine_dataframes(
        left.to_pandas(), right, ignore_names=ignore_names, prefer_left=prefer_left
    )


@dispatch(object, object)
def combine(
    left: object, right: object, ignore_names: bool = False, prefer_left: bool = False
):
    raise ValueError(f"Cannot combine types: {type(left)} and {type(right)}")
