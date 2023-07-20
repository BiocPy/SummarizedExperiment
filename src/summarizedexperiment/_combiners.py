from typing import Optional, Sequence, Tuple, Union, Literal, MutableMapping
from functools import reduce
import pandas as pd
import numpy as np
import scipy.sparse as sp

ArrayTypes = Union[np.ndarray, sp.lil_matrix]

from ._validators import validate_names, validate_shapes
from .dispatchers.combiners import combine, combine_prefer_left, combine_ignore_names


def _impose_common_precision(x: ArrayTypes, y: ArrayTypes):
    """Ensure input arrays have compatible dtypes.

    Args:
        x (ArrayTypes): first array.
        y (ArrayTypes): second array.
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


def combine_concatenation_axis(
    ses: Sequence["BaseSE"], experiment_metadata: Literal["rowData", "colData"]
) -> pd.DataFrame:
    """Method for combining metadata along the concatenation axis.

    Args:
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects.
        experiment_metadata (Literal["rowData", "colData"]): the experiment_metadata to combine.

    Returns:
        concatenated_df (pd.DataFrame): the concatenated experiment metadata.
    """
    all_experiment_metadata = [getattr(se, experiment_metadata) for se in ses]
    return reduce(combine, all_experiment_metadata)


def combine_non_concatenation_axis(
    ses: Sequence["BaseSE"],
    experiment_metadata: Literal["rowData", "colData"],
    useNames: bool,
) -> pd.DataFrame:
    """Method for combining metadata along the non-concatenation axis.

    Args:
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects.
        experiment_metadata (Literal["rowData", "colData"]): the experiment_metadata to combine.
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


def combine_assays_by_column(
    assay_name: str,
    ses: Sequence["BaseSE"],
    names: pd.Index,
    shape: Tuple[int, int],
    useNames: bool,
) -> sp.lil_matrix:
    """Combine assays across all "SummarizedExperiment" objects by column.

    Args:
        assay_name (str): name of the assay.
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects whose assays to combine.
        names (pd.Index): names of the metadata from the non-concatenation axis.
        shape (Tuple[int, int]): shape of the combined assay.
        useNames (bool): see `combineCols()`.

    Returns:
        merged_assays (sp.lil_matrix): a sparse array of the merged assays.
    """
    col_idx = 0
    merged_assays = sp.lil_matrix(shape)
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
                shared_idxs = np.argwhere(names.isin(se.rownames)).squeeze()
                merged_assays[shared_idxs, col_idx : col_idx + offset] = curr_assay
            else:
                merged_assays[:, col_idx : col_idx + offset] = curr_assay
        col_idx += offset

    return merged_assays


def combine_assays_by_row(
    assay_name: str,
    ses: Sequence["BaseSE"],
    names: pd.Index,
    shape: Tuple[int, int],
    useNames: bool,
) -> np.ndarray:
    """Combine assays across all "SummarizedExperiment" objects by row.

    Args:
        assay_name (str): name of the assay.
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects whose assays to combine.
        names (pd.Index): names of the metadata from the non-concatenation axis.
        shape (Tuple[int, int]): shape of the combined assay.
        useNames (bool): see `combineCols()`.
    """
    raise NotImplementedError("combine_assays_by_row not implemented yet")


def combine_assays(
    assay_name: str,
    ses: Sequence["BaseSE"],
    names: pd.Index,
    by: Literal["row", "column"],
    shape: Tuple[int, int],
    useNames: bool,
) -> np.ndarray:
    """Combine assays across all "SummarizedExperiment" objects.

    Args:
        assay_name (str): name of the assay.
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects whose assays to combine.
        names (pd.Index): names of the metadata from the non-concatenation axis.
        by (Literal["row", "column"]): the concatenation axis.
        shape (Tuple[int, int]): shape of the combined assay.
        useNames (bool): see `combineCols()`.
    """
    if by == "row":
        return combine_assays_by_row(
            assay_name=assay_name, ses=ses, names=names, shape=shape, useNames=useNames
        )
    elif by == "column":
        return combine_assays_by_column(
            assay_name=assay_name, ses=ses, names=names, shape=shape, useNames=useNames
        )
    else:
        raise ValueError(f"cannot combine assays by {by}. can only combine by row or column.")
