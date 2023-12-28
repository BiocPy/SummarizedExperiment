from typing import Any

# from .RangedSummarizedExperiment import RangedSummarizedExperiment

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche"
__license__ = "MIT"


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
        x:
            Any object.

    Returns:
        True if ``x``is matrix-like.
    """
    # TODO: this only work for python 3.8 and below.
    # return isinstance(x, MatrixProtocol)
    return hasattr(x, "__getitem__") and hasattr(x, "shape")
