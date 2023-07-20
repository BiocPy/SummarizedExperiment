from typing import Sequence
from functools import reduce
import pandas as pd
import numpy as np

from ._validators import validate_row_names


def combine(rowDatas: Sequence[pd.DataFrame], useNames: bool) -> pd.DataFrame:
    """Combine rowDatas.

    Args:
        rowDatas (pd.DataFrame): rowDatas to combine.
        useNames (bool): whether or not to use index names.

    Returns:
        pd.DataFrame: combined rowDatas.
    """
    if useNames:
        validate_row_names(rowDatas)
    else:
        row_names = rowDatas[0].index
        for rowData in rowDatas[1:]:
            rowData.index = row_names
    return reduce(lambda left, right: left.combine_first(right), rowDatas)


def create_samples_if_missing(sample_names: Sequence[str], df: pd.DataFrame):
    """Create a new sample populated with nans if it doesn't exist.

    Args:
        sample_names (str): List of sample names that should exist.
        df (pd.DataFrame): The dataframe.
    """
    for sample_name in sample_names:
        if sample_name not in df.columns:
            df[sample_name] = np.nan


def create_features_if_missing(
    feature_names: Sequence[str], df: pd.DataFrame
) -> pd.DataFrame:
    """Create a new feature populated with nans if it doesn't exist.

    Args:
        feature_names (str): List of feature names that should exist.
        df (pd.DataFrame): The dataframe.

    Returns:
        pd.DataFrame: Assay with missing features added.
    """
    all_features = df.index.union(feature_names, sort=False)
    return df.reindex(index=all_features)
