from typing import MutableMapping, Optional

from .BaseSE import BaseSE
from .types import BiocOrPandasFrame, MatrixTypes, SlicerArgTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """Container to represents genomic experiment data (`assays`), features (`row_data`),
    sample data (`col_data`) and any other `metadata`.

    SummarizedExperiment follows the R/Bioconductor specification; rows are features, columns
    are samples.

    Attributes:
        assays (MutableMapping[str, MatrixTypes]): Dictionary
            of matrices, with assay names as keys and 2-dimensional matrices represented as
            :py:class:`~numpy.ndarray` or :py:class:`scipy.sparse.spmatrix` matrices.

            Alternatively, you may use any 2-dimensional matrix that contains the property ``shape``
            and implements the slice operation using the ``__getitem__`` dunder method.

            All matrices in ``assays`` must be 2-dimensional and have the same
            shape (number of rows, number of columns).

        row_data (BiocOrPandasFrame, optional): Features, must be the same length as
            rows of the matrices in assays.

            Features may be either a :py:class:`~pandas.DataFrame` or
            :py:class:`~biocframe.BiocFrame.BiocFrame`.

            Defaults to None.
        col_data (BiocOrPandasFrame, optional): Sample data, must be
            the same length as columns of the matrices in assays.

            Sample Information may be either a :py:class:`~pandas.DataFrame` or
            :py:class:`~biocframe.BiocFrame.BiocFrame`.

            Defaults to None.
        metadata (MutableMapping, optional): Additional experimental metadata describing the
            methods. Defaults to None.
    """

    def __init__(
        self,
        assays: MutableMapping[str, MatrixTypes],
        rowData: Optional[BiocOrPandasFrame] = None,
        colData: Optional[BiocOrPandasFrame] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize a Summarized Experiment (SE)."""
        super().__init__(assays, rowData, colData, metadata)

    def __getitem__(
        self,
        args: SlicerArgTypes,
    ) -> "SummarizedExperiment":
        """Subset a `SummarizedExperiment`.

        Args:
            args (SlicerArgTypes): Indices or names to slice. Tuple contains
                slices along dimensions (rows, cols).

                Each element in the tuple, might be either a integer vector (integer positions),
                boolean vector or :py:class:`~slice` object. Defaults to None.

        Raises:
            ValueError: Too many or few slices.

        Returns:
            SummarizedExperiment: Sliced `SummarizedExperiment` object.
        """
        sliced_objs = self._slice(args)
        return SummarizedExperiment(
            assays=sliced_objs.assays,
            rowData=sliced_objs.rowData,
            colData=sliced_objs.colData,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        pattern = (
            f"Class SummarizedExperiment with {self.shape[0]} features and {self.shape[1]} "
            "samples \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  features: {self.rowData.columns if self.rowData is not None else None} \n"
            f"  sample data: {self.colData.columns if self.colData is not None else None}"
        )
        return pattern
