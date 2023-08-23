from typing import Any, Callable

import pandas as pd
from biocframe import BiocFrame

# from .RangedSummarizedExperiment import RangedSummarizedExperiment

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche"
__license__ = "MIT"


def is_bioc_or_pandas_frame(x: Any) -> bool:
    """Checks if ``x`` is either a :py:class:`~pandas.DataFrame` or :py:class:`~biocframe.BiocFrame.BiocFrame`.

    Args:
        x (Any): Any object.

    Returns:
        bool: True if ``x`` is `DataFrame`-like.
    """
    return isinstance(x, pd.DataFrame) or isinstance(x, BiocFrame)


# def is_gr_or_rse(x: Union[GenomicRanges, RangedSummarizedExperiment]):
#     """Check if the object is either a `RangedSummarizedExperiment` or `GenomicRanges`.

#     Args:
#         x (Union[GenomicRanges, RangedSummarizedExperiment]): object to check.

#     Raises:
#         TypeError: object is not a `RangedSummarizedExperiment` or `GenomicRanges`.
#     """
#     if not (isinstance(x, RangedSummarizedExperiment) or isinstance(x, GenomicRanges)):
#         raise TypeError(
#             "object is not a `RangedSummarizedExperiment` or `GenomicRanges`"
#         )


# def _get_python_minor():
#     return version_info[1] < 8
# Expectations on Assays, these should be matrices or matrix-like objects
# that support slicing and expose a shape parameter.
# TODO: this only works for Python 3.8 and above.
#
# if _get_python_minor() > 8:
#     @runtime_checkable
#     class MatrixProtocol(Protocol):
#         def __getitem__(self, args):
#             ...

#         @property
#         def shape(self):
#             ...


def is_matrix_like(x: Any) -> bool:
    """Check if ``x`` is a `matrix`-like object.

    Matrix must support the matrix protocol, has the `shape` property
    and allows slicing by implementing the `__getitem__` dunder method.

    Args:
        x (Any): Any object.

    Returns:
        bool: True if ``x``is matrix-like.
    """
    # TODO: this only work for python 3.8 and below.
    # return isinstance(x, MatrixProtocol)
    return hasattr(x, "__getitem__") and hasattr(x, "shape")


def is_list_of_type(x: Any, target_type: Callable) -> bool:
    """Checks if ``x`` is a list or tuple and and whether all elements are of the same type.

    Args:
        x (Any): Any object.
        target_type (callable): Type to check for, e.g. ``str``, ``int``.

    Returns:
        bool: True if ``x`` is :py:class:`~list` and all elements are of the same type.
    """
    return (isinstance(x, list) or isinstance(x, tuple)) and all(
        isinstance(item, target_type) for item in x
    )


def is_list_of_subclass(x: Any, target_type: Callable) -> bool:
    """Checks if all provided objects subclass of ``target_type``.

    Args:
        x (Any): Any object.
        target_type (callable): Type to check objects against.

    Returns:
        bool: True if ``x`` is :py:class:`~list` and all objects are derivatives of the same class.
    """
    return (isinstance(x, list) or isinstance(x, tuple)) and all(
        issubclass(type(item), target_type) for item in x
    )
