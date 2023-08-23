from typing import Dict, List, Literal, Sequence, Tuple

from biocframe import BiocFrame
from numpy import argwhere, find_common_type, ndarray
from pandas import DataFrame, Index, concat
from scipy.sparse import lil_matrix

from ..types import ArrayTypes, BiocOrPandasFrame
from .validators import validate_names, validate_shapes
from ..dispatchers import get_rownames

# from ..SummarizedExperiment import SummarizedExperiment

__author__ = "keviny2, jkanche"
__copyright__ = "keviny2"
__license__ = "MIT"


def impose_common_precision(*x: ArrayTypes) -> List[ArrayTypes]:
    """Check and transform input arrays into common dtypes.

    Args:
        *x (ArrayTypes): Array like objects.

            ``x`` may be either a :py:class:`~scipy.sparse.lil_matrix` or
            a :py:class:`~numpy.ndarray`.

    Returns:
        List[ArrayTypes]: All transformed matrices.
    """
    common_dtype = find_common_type([m.dtype for m in x], [])
    return [(m.astype(common_dtype) if m.dtype != common_dtype else m) for m in x]


def _remove_duplicate_columns(df: DataFrame) -> DataFrame:
    """Remove duplicate columns from a :py:class:`pandas.DataFrame`.

    Args:
        df (DataFrame): Input `DataFrame` with possible duplicate columns.

    Returns:
        DataFrame: A new DataFrame with duplicate columns removed.
    """
    return df.loc[:, ~df.columns.duplicated()]


def combine_metadata(experiments: Sequence["SummarizedExperiment"]) -> Dict:
    """Combine metadata across experiments.

    Args:
        experiments (Sequence[SummarizedExperiment]): `SummarizedExperiment`-like objects.

    Returns:
        Dict: A dictionary with combined metadata across all input ``experiments``.
    """
    combined_metadata = {}
    for i, se in enumerate(experiments):
        if se.metadata is not None:
            combined_metadata[i] = se.metadata

    return combined_metadata


def combine_frames(
    x: Sequence[BiocOrPandasFrame],
    use_names: bool,
    axis: int,
    remove_duplicate_columns: bool,
) -> DataFrame:
    """Combine a series of `DataFrame`-like objects.

    Args:
        x (Sequence[BiocOrPandasFrame]): Input frames.
            May be either a :py:class:`~biocframe.BiocFrame.BiocFrame` or
            :py:class:`~pandas.DataFrame`.

        use_names (bool): ``True`` to use index names to merge.
            If ``False``, merges pair-wise along an axis.

        axis (int): Axis to merge on, 0 for rows, 1 for columns.

        remove_duplicate_columns (bool): If `True`, remove any duplicate columns in
            `row_data` or `col_data` of the resultant
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

    Returns:
        DataFrame: A merged data frame.
    """
    all_as_pandas = [(m.to_pandas() if isinstance(m, BiocFrame) else m) for m in x]

    if use_names is True:
        validate_names(all_as_pandas)
    else:
        validate_shapes(all_as_pandas)
        # reset names
        all_as_pandas = [df.reset_index(drop=True) for df in all_as_pandas]

    concat_df = concat(all_as_pandas, axis=axis)

    if (use_names is False) and (get_rownames(x[0]) is not None):
        concat_df.index = get_rownames(x[0])

    if remove_duplicate_columns:
        return _remove_duplicate_columns(concat_df)

    return concat_df


def combine_assays_by_column(
    assay_name: str,
    experiments: Sequence["SummarizedExperiment"],
    names: Index,
    shape: Tuple[int, int],
    use_names: bool,
) -> lil_matrix:
    """Combine :py:attr:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.assays` across all
    :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` objects by **column**.

    Args:
        assay_name (str): Name of the assay.
        experiments (Sequence[SummarizedExperiment]):
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` objects to
            combine.
        names (Index): Names of the metadata from the non-concatenation axis.
        shape (Tuple[int, int]): Shape of the combined assay.
        use_names (bool): See
            :py:meth:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.combine_cols`.

    Returns:
        lil_matrix: A sparse array of the merged assays.
    """
    col_idx = 0
    merged_assays = lil_matrix(shape)
    for i, se in enumerate(experiments):
        offset = se.shape[1]
        if assay_name not in se.assays:
            merged_assays[
                :, col_idx : col_idx + offset
            ] = 0  # do we want to fill with nan or 0's?
        else:
            curr_assay = se.assays[assay_name]
            impose_common_precision(merged_assays, curr_assay)
            if use_names:
                shared_idxs = argwhere(names.isin(se.row_names)).squeeze()
                merged_assays[shared_idxs, col_idx : col_idx + offset] = curr_assay
            else:
                merged_assays[:, col_idx : col_idx + offset] = curr_assay

        col_idx += offset

    return merged_assays


def combine_assays_by_row(
    assay_name: str,
    experiments: Sequence["SummarizedExperiment"],
    names: Index,
    shape: Tuple[int, int],
    use_names: bool,
) -> ndarray:
    """Combine assays across all :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` objects by
    **row**.

    Args:
        assay_name (str): Name of the assay.
        experiments (Sequence[SummarizedExperiment]):
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` objects.
            to combine.
        names (Index): Names of the metadata from the non-concatenation axis.
        shape (Tuple[int, int]): Shape of the combined assay.
        use_names (bool): See
            :meth:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.combine_cols`.
    """
    raise NotImplementedError("`combine_assays_by_row` not implemented yet")


def combine_assays(
    assay_name: str,
    experiments: Sequence["SummarizedExperiment"],
    names: Index,
    by: Literal["row", "column"],
    shape: Tuple[int, int],
    use_names: bool,
) -> lil_matrix:
    """Combine assays across all :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` objects.

    Args:
        assay_name (str): Name of the assay.
        experiments (Sequence[BaseSE]):
            :py:class:`summarizedexperiment.SummarizedExperiment.SummarizedExperiment` objects.
        names (Index): Names of the metadata from the non-concatenation axis.
        by (Literal["row", "column"]): Concatenation axis.
        shape (Tuple[int, int]): Shape of the combined assay.
        use_names (bool): See
            :py:meth:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.combine_cols`.

    Returns:
        lil_matrix: A sparse array of the merged assays.
    """
    if by == "row":
        return combine_assays_by_row(
            assay_name=assay_name,
            experiments=experiments,
            names=names,
            shape=shape,
            use_names=use_names,
        )
    elif by == "column":
        return combine_assays_by_column(
            assay_name=assay_name,
            experiments=experiments,
            names=names,
            shape=shape,
            use_names=use_names,
        )
    else:
        raise ValueError(
            f"Cannot combine assays by {by}. can only combine by row or column."
        )
