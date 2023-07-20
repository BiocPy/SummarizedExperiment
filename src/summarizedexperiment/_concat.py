from typing import Optional, Sequence, Tuple, Union, Literal, MutableMapping
import pandas as pd
import numpy as np
from biocframe import BiocFrame


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

def concatenate(ses: Sequence["BaseSE"], property: Literal["rowData", "colData"]):
    """Concatenate along the concatenation axis.
    
    Args:
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects.
        property (Literal["rowData", "colData"]): the property to concatenate along.
    """
    return pd.concat([getattr(se, property) for se in ses])

def impose_common_precision(x: np.ndarray, y: np.ndarray):
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
            impose_common_precision(merged_assays, curr_assay)
            if useNames:
                shared_idxs = np.argwhere(other.index.isin(se.rownames)).squeeze()
                merged_assays[shared_idxs, col_idx : col_idx + offset] = curr_assay
            else:
                merged_assays[:, col_idx : col_idx + offset] = curr_assay
        col_idx += offset

    return merged_assays
