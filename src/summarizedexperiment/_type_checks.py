from typing import Any, Protocol, runtime_checkable

import pandas as pd
from biocframe import BiocFrame

# from .RangeSummarizedExperiment import RangeSummarizedExperiment

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche"
__license__ = "MIT"


def is_bioc_or_pandas_frame(x: Any) -> bool:
    """Checks if `x` is either a Pandas DataFrame or BiocFrame.

    Args:
        x (Any): any object.

    Returns:
        bool: True if `x` is dataframe-ish.
    """
    return isinstance(x, pd.DataFrame) or isinstance(x, BiocFrame)


# def is_gr_or_rse(x: Union[GenomicRanges, RangeSummarizedExperiment]):
#     """Check if the object is either a `RangeSummarizedExperiment` or `GenomicRanges`.

#     Args:
#         x (Union[GenomicRanges, RangeSummarizedExperiment]): object to check.

#     Raises:
#         TypeError: object is not a `RangeSummarizedExperiment` or `GenomicRanges`.
#     """
#     if not (isinstance(x, RangeSummarizedExperiment) or isinstance(x, GenomicRanges)):
#         raise TypeError(
#             "object is not a `RangeSummarizedExperiment` or `GenomicRanges`"
#         )


# Expectations on Assays, these should be matrices or matrix-like objects
# that support slicing and expose a shape parameter.
@runtime_checkable
class MatrixProtocol(Protocol):
    def __getitem__(self, args):
        ...

    @property
    def shape(self):
        ...


def is_matrix_like(x: Any) -> bool:
    """Check if `x` is a matrix-like object.

    Matrix must support the matrix protocol, has the `shape` property
    and allows slicing.

    Args:
        x (Any): any object.

    Returns:
        bool: True if it matrix-like.
    """
    return isinstance(x, MatrixProtocol)


def is_list_of_type(x: Any, type: callable) -> bool:
    """Checks if `x` is a list of booleans.

    Args:
        x (Any): any object.
        type (callable): Type to check for, e.g. str, int

    Returns:
        bool: True if `x` is list and all values are of the same type.
    """
    return isinstance(x, list) and all(isinstance(item, type) for item in x)
