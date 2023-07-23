from typing import Literal, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from biocframe import BiocFrame

from ._types import ArrayTypes, BiocOrPandasFrame
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


def combine_metadata(experiments: Sequence["BaseSE"]) -> MutableMapping:
    """Combine metadata across experiments.

    Args:
        experiments (Sequence[BaseSE]): "SummarizedExperiment"-like objects.

    Returns:
        MutableMapping: combined metadata.
    """
    combined_metadata = {}
    for i, se in enumerate(experiments):
        if se.metadata:
            combined_metadata[i] = se.metadata

    return combined_metadata


def combine_frames(
    x: Sequence[BiocOrPandasFrame], useNames: bool, axis: int
) -> pd.DataFrame:
    """Combine a Bioc or Pandas dataframe.

    Args:
        x (Sequence[BiocOrPandasFrame]): input frames.
        useNames (bool): use index names to merge? Otherwise merges
            pair-wise along an axis.
        axis (int): axis to merge on, 0 for rows, 1 for columns.

    Returns:
        pd.DataFrame: merged data frame
    """
    all_as_pandas = [(m.to_pandas() if isinstance(x, BiocFrame) else m) for m in x]

    if useNames is True:
        validate_names(all_as_pandas)
    elif axis == 0:
        validate_shapes(all_as_pandas)

    return pd.concat(all_as_pandas, ignore_index=(not useNames), axis=axis)


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
) -> sp.lil_matrix:
    """Combine assays across all "SummarizedExperiment" objects.

    Args:
        assay_name (str): name of the assay.
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects whose assays to combine.
        names (pd.Index): names of the metadata from the non-concatenation axis.
        by (Literal["row", "column"]): the concatenation axis.
        shape (Tuple[int, int]): shape of the combined assay.
        useNames (bool): see `combineCols()`.

    Returns:
        merged_assays (sp.lil_matrix): a sparse array of the merged assays.
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
        raise ValueError(
            f"cannot combine assays by {by}. can only combine by row or column."
        )
