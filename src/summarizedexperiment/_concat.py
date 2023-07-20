from typing import Optional, Sequence, Tuple, Union, Literal, MutableMapping
from functools import reduce
import pandas as pd
import numpy as np
from biocframe import BiocFrame

from ._validators import validate_names, validate_shapes
from .dispatchers.combine import combine, combine_prefer_left, combine_ignore_names


def _drop_index(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the indexes of a dataframe.

    Args:
        df (pd.DataFrame): a dataframe.

    Returns:
        df (pd.DataFrame): a dataframe with indexes removed.
    """
    return df.reset_index(drop=True)


def _merge_ignore_names(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge dataframes ignoring names.

    Args:
        dfs (Sequence[pd.DataFrame]): dataframes to merge.

    Returns:
        merged_df (pd.DataFrame): the merged dataframe.
    """
    if len(dfs) == 1:
        return _drop_index(dfs[0])

    return pd.concat(
        [_drop_index(dfs[0]), _merge_ignore_names(dfs[1:])], axis=1, join="inner"
    )


def _impose_common_precision(x: np.ndarray, y: np.ndarray):
    """Ensure input arrays have compatible dtypes.

    Args:
        x (np.ndarray): first array.
        y (np.ndarray): second array.
    """
    dtype = np.find_common_type([x.dtype, y.dtype], [])
    if x.dtype != dtype:
        x = x.astype(dtype)
    if y.dtype != dtype:
        y = y.astype(dtype)


def combine_metadata(ses: Sequence["BaseSE"]) -> Optional[MutableMapping]:
    """Combine metadata across experiments.

    Args:
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects.

    Returns:
        combined_metadata (Optional[MutableMapping]): combined metadata.
    """
    combined_metadata = []
    for se in ses:
        if se.metadata:
            combined_metadata.extend(se.metadata.values())
    return dict(enumerate(combined_metadata))


def concatenate(
    ses: Sequence["BaseSE"], experiment_metadata: Literal["rowData", "colData"]
) -> pd.DataFrame:
    """Concatenate along the concatenation axis.

    Args:
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects.
        experiment_metadata (Literal["rowData", "colData"]): the experiment_metadata to concatenate along.

    Returns:
        concatenated_df (pd.DataFrame): the concatenated experiment metadata.
    """
    all_experiment_metadata = [getattr(se, experiment_metadata) for se in ses]
    return reduce(combine, all_experiment_metadata)


def concatenate_other(
    ses: Sequence["BaseSE"],
    experiment_metadata: Literal["rowData", "colData"],
    useNames: bool,
) -> pd.DataFrame:
    """Concatenate along the non-concatenation axis, ignoring names.

    Args:
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects.
        experiment_metadata (Literal["rowData", "colData"]): the experiment_metadata to concatenate along.
        useName (bool): see `combineCols()`

    Returns:
        concatenated_df (pd.DataFrame): the concatenated experiment metadata.
    """
    all_experiment_metadata = [getattr(se, experiment_metadata) for se in ses]
    if useNames:
        validate_names(ses, experiment_metadata=experiment_metadata)
        return reduce(combine_prefer_left, all_experiment_metadata)
    else:
        validate_shapes(ses, experiment_metadata=experiment_metadata)
        names = getattr(ses[0], experiment_metadata).index
        return reduce(combine_ignore_names, all_experiment_metadata).set_index(names)


def combine_assays(
    assay_name: str,
    ses: Sequence["BaseSE"],
    other: Union[pd.DataFrame, BiocFrame],
    shape: Tuple[int, int],
    useNames: bool,
) -> np.ndarray:
    """Combine assays across all "SummarizedExperiment" objects.

    Args:
        assay_name (str): name of the assay.
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects whose assays to combine.
        other (Union[pd.DataFrame, BiocFrame]): object from the non-concatenation axis.
        shape (Tuple[int, int]): shape of the combined assay.
        useNames (bool): see `combineCols()`.
    """
    col_idx = 0
    merged_assays = np.zeros(shape=shape)
    for se in ses:
        offset = se.shape[1]
        if assay_name not in se.assays:
            merged_assays[
                :, col_idx : col_idx + offset
            ] = 0  # do we want to fill with np.nan or 0's?
        else:
            curr_assay = se.assays[assay_name]
            _impose_common_precision(merged_assays, curr_assay)
            if useNames:
                shared_idxs = np.argwhere(other.index.isin(se.rownames)).squeeze()
                merged_assays[shared_idxs, col_idx : col_idx + offset] = curr_assay
            else:
                merged_assays[:, col_idx : col_idx + offset] = curr_assay
        col_idx += offset

    return merged_assays
