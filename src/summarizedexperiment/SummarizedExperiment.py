from typing import MutableMapping, Optional

from .BaseSE import BaseSE
from .types import BiocOrPandasFrame, MatrixTypes, SlicerArgTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """SummarizedExperiment class to represent genomic experiment data,
    features, sample data and any other metadata.
    """

    def __init__(
        self,
        assays: MutableMapping[str, MatrixTypes],
        rowData: Optional[BiocOrPandasFrame] = None,
        colData: Optional[BiocOrPandasFrame] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize a Summarized Experiment (SE).

        SE follows the R/Bioconductor specification; rows are features, columns are
        samples.

        Args:
            assays (MutableMapping[str, MatrixTypes]): dictionary
                of matrices, with assay names as keys and matrices represented as dense
                (numpy) or sparse (scipy) matrices. All matrices across assays must
                have the same dimensions (number of rows, number of columns).
            rowData (BiocOrPandasFrame, optional): features, must be the same length as
                rows of the matrices in assays. Defaults to None.
            colData (BiocOrPandasFrame, optional): sample data, must be
                the same length as columns of the matrices in assays. Defaults to None.
            metadata (MutableMapping, optional): experiment metadata describing the
                methods. Defaults to None.
        """
        super().__init__(assays, rowData, colData, metadata)

    def __getitem__(
        self,
        args: SlicerArgTypes,
    ) -> "SummarizedExperiment":
        """Subset a `SummarizedExperiment`.

        Args:
            args (SlicerArgTypes): indices to slice.
                tuple can contains slices along dimensions (rows, cols).

        Raises:
            ValueError: Too many or few slices.

        Returns:
            SummarizedExperiment: sliced `SummarizedExperiment` object.
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
            f"Class SummarizedExperiment with {self.shape[0]} features and {self.shape[1]} samples \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  features: {self.rowData.columns if self.rowData is not None else None} \n"
            f"  sample data: {self.colData.columns if self.colData is not None else None}"
        )
        return pattern
