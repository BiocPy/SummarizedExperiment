from functools import reduce
from typing import Literal, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._types import ArrayTypes
from ._validators import validate_experiment_attribute, validate_names, validate_shapes
from ..dispatchers.combiners import combine

__author__ = "keviny2"
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
    ses: Sequence["BaseSE"], experiment_attribute: Literal["rowData", "colData"]
) -> pd.DataFrame:
    """Method for combining metadata along the concatenation axis.

    Args:
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects.
        experiment_attribute (Literal["rowData", "colData"]): the experiment_attribute to combine.

    Returns:
        concatenated_df (pd.DataFrame): the concatenated experiment metadata.
    """
    validate_experiment_attribute(experiment_attribute=experiment_attribute)

    all_experiment_attributes = [getattr(se, experiment_attribute) for se in ses]
    return reduce(combine, all_experiment_attributes)


def combine_non_concatenation_axis(
    ses: Sequence["BaseSE"],
    experiment_attribute: Literal["rowData", "colData"],
    useNames: bool,
) -> pd.DataFrame:
    """Method for combining metadata along the non-concatenation axis.

    Args:
        ses (Sequence[BaseSE]): "SummarizedExperiment" objects.
        experiment_attribute (Literal["rowData", "colData"]): the experiment_attribute to combine.
        useName (bool): see `combineCols()`

    Returns:
        concatenated_df (pd.DataFrame): the concatenated experiment metadata.
    """
    validate_experiment_attribute(experiment_attribute=experiment_attribute)

    all_experiment_attributes = [getattr(se, experiment_attribute) for se in ses]
    if useNames:
        validate_names(ses, experiment_attribute=experiment_attribute)
        return reduce(
            lambda left, right: combine(left, right, prefer_left=True),
            all_experiment_attributes,
        )
    else:
        validate_shapes(ses, experiment_attribute=experiment_attribute)
        combined_dataframe = reduce(
            lambda left, right: combine(left, right, ignore_names=True),
            all_experiment_attributes,
        )
        names = getattr(ses[0], experiment_attribute).index
        combined_dataframe.index = names
        return combined_dataframe


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
