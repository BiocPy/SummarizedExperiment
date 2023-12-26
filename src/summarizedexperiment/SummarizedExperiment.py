from typing import Dict, List, Optional
from warnings import warn

from biocframe import BiocFrame
from genomicranges import GenomicRanges

from .BaseSE import BaseSE
from .types import MatrixTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """Container to represents genomic experiment data (`assays`), features (`row_data`), sample data (`column_data`) and
    any other `metadata`.

    SummarizedExperiment follows the R/Bioconductor specification; rows are features, columns are samples.
    """

    def __init__(
        self,
        assays: Dict[str, MatrixTypes],
        row_data: Optional[BiocFrame] = None,
        column_data: Optional[BiocFrame] = None,
        row_names: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        validate: bool = True,
    ) -> None:
        """Initialize a Summarized Experiment (SE).

        Args:
            assays:
                A dictionary containing matrices, with assay names as keys
                and 2-dimensional matrices represented as either
                :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

                Alternatively, you may use any 2-dimensional matrix that has
                the ``shape`` property and implements the slice operation
                using the ``__getitem__`` dunder method.

                All matrices in assays must be 2-dimensional and have the
                same shape (number of rows, number of columns).

            row_data:
                Features, must be the same length as the number of rows of
                the matrices in assays.

                Feature information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            column_data:
                Sample data, must be the same length as the number of
                columns of the matrices in assays.

                Sample information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            row_names:
                A list of strings, same as the number of rows.Defaults to None.

            column_names:
                A list of string, same as the number of columns. Defaults to None.

            metadata:
                Additional experimental metadata describing the methods.
                Defaults to None.

            validate:
                Internal use only.
        """

        if isinstance(row_data, GenomicRanges):
            warn(
                "`row_data` is `GenomicRanges`, consider using `RangeSummarizedExperiment`."
            )

        super().__init__(
            assays,
            row_data=row_data,
            column_data=column_data,
            row_names=row_names,
            column_names=column_names,
            metadata=metadata,
            validate=validate,
        )
