from typing import Any, Dict, Optional
from warnings import warn

from biocframe import BiocFrame
from genomicranges import GenomicRanges

from .BaseSE import BaseSE
from .types import SlicerArgTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """Container to represent genomic experimental data (`assays`), features (`row_data`), sample data (`col_data`) and
    any other `metadata`.

    SummarizedExperiment follows the R/Bioconductor specification; rows are features, columns
    are samples.

    Attributes:
        assays (Dict[str, Any]): A dictionary containing matrices, with assay names as keys
            and 2-dimensional matrices represented as either
            :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

            Alternatively, you may use any 2-dimensional matrix that has the ``shape`` property and
            implements the slice operation using the ``__getitem__`` dunder method.

            All matrices in assays must be 2-dimensional and have the same shape
            (number of rows, number of columns).

        row_data (BiocFrame, optional): Features, must be the same length as the numner of rows of
            the matrices in assays. Features can be either a :py:class:`~pandas.DataFrame` or
            :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

        col_data (BiocFrame, optional): Sample data, which be the same length as the number of
            columns of the matrices in assays. Sample Information can be either a :py:class:`~pandas.DataFrame`
            or :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

        metadata (Dict, optional): Additional experimental metadata describing the methods. Defaults to None.
    """

    def __init__(
        self,
        assays: Dict[str, Any],
        row_data: Optional[BiocFrame] = None,
        col_data: Optional[BiocFrame] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Initialize a Summarized Experiment (SE).

        Args:
            assays (Dict[str, Any]): A dictionary containing matrices, with assay names as keys
                and 2-dimensional matrices represented as either
                :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

                Alternatively, you may use any 2-dimensional matrix that has the ``shape`` property and
                implements the slice operation using the ``__getitem__`` dunder method.

                All matrices in assays must be 2-dimensional and have the same shape
                (number of rows, number of columns).

            row_data (BiocFrame, optional): Features, must be the same length as the numner of rows of
                the matrices in assays. Features can be either a :py:class:`~pandas.DataFrame` or
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            col_data (BiocFrame, optional): Sample data, which be the same length as the number of
                columns of the matrices in assays. Sample Information can be either a :py:class:`~pandas.DataFrame`
                or :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            metadata (Dict, optional): Additional experimental metadata describing the methods. Defaults to None.
        """

        if isinstance(row_data, GenomicRanges):
            warn(
                "`row_data` is `GenomicRanges`, consider using `RangeSummarizedExperiment`."
            )

        super().__init__(assays, row_data, col_data, metadata)

    def __getitem__(
        self,
        args: SlicerArgTypes,
    ) -> "SummarizedExperiment":
        """Subset a `SummarizedExperiment`.

        Args:
            args (SlicerArgTypes): Indices or names to slice. The tuple contains
                slices along dimensions (rows, cols).

                Each element in the tuple may be either an list of indices,
                boolean vector or :py:class:`~slice` object.

        Raises:
            ValueError: If too many or too few slices are provided.

        Returns:
            The same type as caller, with the sliced entries.
        """
        sliced_objs = self._slice(args)

        current_class_const = type(self)
        return current_class_const(
            assays=sliced_objs.assays,
            row_data=sliced_objs.row_data,
            col_data=sliced_objs.col_data,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        current_class_const = type(self)
        pattern = (
            f"Class {current_class_const.__name__} with {self.shape[0]} features and {self.shape[1]} "
            "samples \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  row_data: {self.row_data.columns if self.row_data is not None else None} \n"
            f"  col_data: {self.col_data.columns if self.col_data is not None else None}"
        )
        return pattern
