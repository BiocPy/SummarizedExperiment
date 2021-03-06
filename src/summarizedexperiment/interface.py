from typing import Union, Dict, Any
from genomicranges import GenomicRanges
import numpy as np
from scipy import sparse as sp
import pandas as pd

from .SummarizedExperiment import SummarizedExperiment as se
from .RangeSummarizedExperiment import RangeSummarizedExperiment as rse

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def SummarizedExperiment(
    assays: Dict[str, Union[np.ndarray, sp.spmatrix]],
    rowData: pd.DataFrame = None,
    rowRanges: GenomicRanges = None,
    colData: pd.DataFrame = None,
    metadata: Any = None,
) -> Union[se, rse]:
    """Validates and Creates either a `SummarizedExperiment` (if rows is `DataFrame`)
    or `RangeSummarizedExperiment` (if features is `GenomicRanges`)

    Args:
        assays (Dict[str, Union[np.ndarray, sp.spmatrix]]): dictionary of experiment data
        rowData (pd.DataFrame): row features as pandas dataframe
        rowRanges (GenomicRanges): row features as GenomicRanges
        colData (pd.DataFrame): sample metadata
        metadata (Any): experiment metadata

    Raises:
        Exception: Assay must contain atleast one experiment data matrix

    Returns:
        Union[SummarizedExperiment, RangeSummarizedExperiment]: either SummarizedExperiment or RangeSummarizedExperiment
    """
    cls = se
    _rows = rowData

    # if rowData is None and rowRanges is None:
    #     raise Exception("Must provide either rowData or rowRanges")

    if (
        assays is None
        or (not isinstance(assays, dict))
        or len(assays.keys()) == 0
    ):
        raise Exception(
            f"{assays} must be a dictionary and contain atleast a single numpy/scipy matrix"
        )

    row_lengths = None
    if rowRanges is not None and isinstance(rowRanges, GenomicRanges):
        cls = rse
        row_lengths = len(rowRanges)
        _rows = rowRanges
    elif rowData is not None:
        row_lengths = rowData.shape[0]

    if colData is not None:
        col_lengths = colData.shape[0]

    # make sure all matrices are the same shape

    matrix_lengths = None
    for d in assays:
        if matrix_lengths is None:
            matrix_lengths = assays[d].shape

        if matrix_lengths != assays[d].shape:
            raise Exception(
                f"matrix dimensions don't match across assays: {d}"
            )

    # are rows same length ?
    if rowData is not None and row_lengths != matrix_lengths[0]:
        raise Exception(
            f"matrix dimensions does not match rowData/rowRanges: {row_lengths} :: {matrix_lengths[0]}"
        )

    # are cols same length ?
    if colData is not None and col_lengths != matrix_lengths[1]:
        raise Exception(
            f"matrix dimensions does not match rowData/rowRanges: {col_lengths} :: {matrix_lengths[1]}"
        )

    return cls(assays=assays, rows=_rows, cols=colData, metadata=metadata)
