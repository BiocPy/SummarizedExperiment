from typing import Literal, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from biocframe import BiocFrame

from ..types import ArrayTypes, BiocOrPandasFrame
from ._validators import validate_names, validate_shapes

__author__ = "keviny2, jkanche"
__copyright__ = "keviny2"
__license__ = "MIT"


def _impose_common_precision(*x: ArrayTypes) -> Sequence[ArrayTypes]:
    """Check and tranform input arrays into common dtypes.

    Args:
        *x (ArrayTypes): array like objects (either sparse or dense).

    Returns:
        Sequence[ArrayTypes]: all transformed matrices
    """
    common_dtype = np.find_common_type([m.dtype for m in x], [])
    return [(m.astype(common_dtype) if m.dtype != common_dtype else m) for m in x]


def _remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with possible duplicate columns.

    Returns:
        pd.DataFrame: A new DataFrame with duplicate columns removed.
    """
    return df.loc[:, ~df.columns.duplicated()]


def combine_metadata(experiments: Sequence["BaseSE"]) -> MutableMapping:
    """Combine metadata across experiments.

    Args:
        experiments (Sequence[BaseSE]): "SummarizedExperiment"-like objects.

    Returns:
        MutableMapping: combined metadata.
    """
    combined_metadata = {}
    for i, se in enumerate(experiments):
        if se.metadata is not None:
            combined_metadata[i] = se.metadata

    return combined_metadata


def combine_frames(
    x: Sequence[BiocOrPandasFrame],
    useNames: bool,
    axis: int,
    removeDuplicateColumns: bool,
) -> pd.DataFrame:
    """Combine a Bioc or Pandas dataframe.

    Args:
        x (Sequence[BiocOrPandasFrame]): input frames.
        useNames (bool): use index names to merge? Otherwise merges
            pair-wise along an axis.
        axis (int): axis to merge on, 0 for rows, 1 for columns.
        removeDuplicateColumns (bool): If `True`, remove any duplicate columns in
            `rowData` or `colData` of the resultant `SummarizedExperiment`.

    Returns:
        pd.DataFrame: merged data frame
    """
    all_as_pandas = [(m.to_pandas() if isinstance(m, BiocFrame) else m) for m in x]

    if useNames is True:
        validate_names(all_as_pandas)
    else:
        validate_shapes(all_as_pandas)
        # reset names
        all_as_pandas = [df.reset_index(drop=True) for df in all_as_pandas]

    concat_df = pd.concat(all_as_pandas, axis=axis)

    if (useNames is False) and (x[0].index is not None):
        concat_df.index = x[0].index

    if removeDuplicateColumns:
        return _remove_duplicate_columns(concat_df)

    return concat_df


def combine_assays_by_column(
    assay_name: str,
    experiments: Sequence["BaseSE"],
    names: pd.Index,
    shape: Tuple[int, int],
    useNames: bool,
) -> sp.lil_matrix:
    """Combine assays across all "SummarizedExperiment" objects by column.

    Args:
        assay_name (str): name of the assay.
        experiments (Sequence[BaseSE]): "SummarizedExperiment" objects whose assays to combine.
        names (pd.Index): names of the metadata from the non-concatenation axis.
        shape (Tuple[int, int]): shape of the combined assay.
        useNames (bool): see `combineCols()`.

    Returns:
        sp.lil_matrix: a sparse array of the merged assays.
    """
    col_idx = 0
    merged_assays = sp.lil_matrix(shape)
    for i, se in enumerate(experiments):
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
    experiments: Sequence["BaseSE"],
    names: pd.Index,
    shape: Tuple[int, int],
    useNames: bool,
) -> np.ndarray:
    """Combine assays across all "SummarizedExperiment" objects by row.

    Args:
        assay_name (str): name of the assay.
        experiments (Sequence[BaseSE]): "SummarizedExperiment" objects whose assays to combine.
        names (pd.Index): names of the metadata from the non-concatenation axis.
        shape (Tuple[int, int]): shape of the combined assay.
        useNames (bool): see `combineCols()`.
    """
    raise NotImplementedError("combine_assays_by_row not implemented yet")


def combine_assays(
    assay_name: str,
    experiments: Sequence["BaseSE"],
    names: pd.Index,
    by: Literal["row", "column"],
    shape: Tuple[int, int],
    useNames: bool,
) -> sp.lil_matrix:
    """Combine assays across all "SummarizedExperiment" objects.

    Args:
        assay_name (str): name of the assay.
        experiments (Sequence[BaseSE]): "SummarizedExperiment" objects whose assays to combine.
        names (pd.Index): names of the metadata from the non-concatenation axis.
        by (Literal["row", "column"]): the concatenation axis.
        shape (Tuple[int, int]): shape of the combined assay.
        useNames (bool): see `combineCols()`.

    Returns:
        sp.lil_matrix: a sparse array of the merged assays.
    """
    if by == "row":
        return combine_assays_by_row(
            assay_name=assay_name,
            experiments=experiments,
            names=names,
            shape=shape,
            useNames=useNames,
        )
    elif by == "column":
        return combine_assays_by_column(
            assay_name=assay_name,
            experiments=experiments,
            names=names,
            shape=shape,
            useNames=useNames,
        )
    else:
        raise ValueError(
            f"cannot combine assays by {by}. can only combine by row or column."
        )
