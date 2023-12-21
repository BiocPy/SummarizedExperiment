from typing import Dict, Optional, Union, Sequence
from warnings import warn

from genomicranges import GenomicRanges
from biocframe import BiocFrame

from .BaseSE import BaseSE
from .types import MatrixTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """Container to represents genomic experiment data (`assays`), 
    features (`row_data`), sample data (`col_data`) and any other `metadata`.

    SummarizedExperiment follows the R/Bioconductor specification; 
    rows are features, columns are samples.
    """

    def __init__(
        self,
        assays: Dict[str, MatrixTypes],
        row_data: Optional[BiocFrame] = None,
        col_data: Optional[BiocFrame] = None,
        metadata: Optional[Dict] = None,
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

            col_data:
                Sample data, must be the same length as the number of
                columns of the matrices in assays.

                Sample information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

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
            assays, row_data=row_data, col_data=col_data, metadata=metadata, validate=validate
        )

    # def __getitem__(
    #     self,
    #     args: Union[int, str, Sequence, tuple],
    # ) -> "SummarizedExperiment":
    #     """Subset a `SummarizedExperiment`.

    #     Args:
    #         args: 
    #             Indices or names to slice. The tuple contains
    #             slices along dimensions (rows, cols).

    #             Each element in the tuple, might be either a integer vector (integer positions),
    #             boolean vector or :py:class:`~slice` object. Defaults to None.

    #     Raises:
    #         ValueError: 
    #             If too many or too few slices provided.

    #     Returns:
    #         Sliced `SummarizedExperiment` object.
    #     """
    #     sliced_objs = self._generic_slice(args)
    #     return SummarizedExperiment(
    #         assays=sliced_objs.assays,
    #         row_data=sliced_objs.rows,
    #         col_data=sliced_objs.cols,
    #         metadata=self.metadata,
    #     )

    def __repr__(self) -> str:
        pattern = (
            f"Class SummarizedExperiment with {self.shape[0]} features and {self.shape[1]} "
            "samples \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  row_data: {self.row_data.columns if self.row_data is not None else None} \n"
            f"  col_data: {self.col_data.columns if self.col_data is not None else None}"
        )
        return pattern
