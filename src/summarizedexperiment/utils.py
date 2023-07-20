from typing import Sequence
from functools import reduce
import numpy as np
import pandas as pd
from pandas import DataFrame


def combine(dfs: Sequence[DataFrame]) -> DataFrame:
    """Combine DataFrames.

    Args:
        dfs (DataFrame): DataFrames to combine.

    Returns:
        DataFrame: combined DataFrame.
    """
    # Bioc gives a warning if similar columns don't have same values -
    # could we use pd.combine to do this?
    return reduce(lambda left, right: left.combine_first(right), dfs)


def create_samples_if_missing(sample_names: Sequence[str], df: DataFrame):
    """Create a new sample populated with nans if it doesn't exist.

    Args:
        sample_names (str): List of sample names that should exist.
        df (DataFrame): The dataframe.
    """
    for sample_name in sample_names:
        if sample_name not in df.columns:
            df[sample_name] = np.nan


def create_features_if_missing(
    feature_names: Sequence[str], df: DataFrame
) -> DataFrame:
    """Create a new feature populated with nans if it doesn't exist.

    Args:
        feature_names (str): List of feature names that should exist.
        df (DataFrame): The dataframe.

    Returns:
        DataFrame: Assay with missing features added.
    """
    all_features = df.index.union(feature_names, sort=False)
    return df.reindex(index=all_features)
