from typing import Sequence
from functools import reduce
import pandas as pd
import numpy as np

from ._validators import validate_names, validate_shapes


def blend(dfs: Sequence[pd.DataFrame], useNames: bool) -> pd.DataFrame:
    """Blend DataFrames.

    Args:
        dfs (pd.DataFrame): DataFrames to blend.
        useNames (bool): whether or not to use names.

    Returns:
        pd.DataFrame: blended DataFrame objects.
    """
    if useNames:
        validate_names(dfs)
    else:
        validate_shapes(dfs)
        names = dfs[0].index
        for df in dfs[1:]:
            df.index = names
    return reduce(lambda left, right: left.combine_first(right), dfs)


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
