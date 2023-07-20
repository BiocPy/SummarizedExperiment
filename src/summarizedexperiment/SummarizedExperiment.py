from typing import Union, Tuple, Optional, Sequence, MutableMapping
from biocframe import BiocFrame
import pandas as pd
import numpy as np
from scipy import sparse as sp

from .BaseSE import BaseSE

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """SummarizedExperiment class to represent genomic experiment data,
    features, sample data and any other metadata.
    """

    def __init__(
        self,
        assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
        rowData: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        colData: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize a Summarized Experiment (SE).

        SE follows the R/Bioconductor specification; rows are features, columns are 
        samples.

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): dictionary 
                of matrices, with assay names as keys and matrices represented as dense 
                (numpy) or sparse (scipy) matrices. All matrices across assays must 
                have the same dimensions (number of rows, number of columns).
            rowData (GenomicRanges, optional): features, must be the same length as 
                rows of the matrices in assays. Defaults to None.
            colData (Union[pd.DataFrame, BiocFrame], optional): sample data, must be 
                the same length as rows of the matrices in assays. Defaults to None.
            metadata (MutableMapping, optional): experiment metadata describing the 
                methods. Defaults to None.
        """
        super().__init__(assays, rowData, colData, metadata)

    @property
    def rowData(self) -> Union[pd.DataFrame, BiocFrame]:
        """Get features.

        Returns:
            Optional[Union[pd.DataFrame, BiocFrame]]: features information.
        """
        return self._rows

    @rowData.setter
    def rowData(self, rows: Union[pd.DataFrame, BiocFrame]) -> None:
        """Set features.

        Args:
            rows (Optional[Union[pd.DataFrame, BiocFrame]]): new feature information.
        """
        rows = rows if rows is not None else BiocFrame({}, numberOfRows=self._shape[0])
        self._validate_rows(rows)
        self._rows = rows

    def __getitem__(
        self,
        args: Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]],
    ) -> "SummarizedExperiment":
        """Subset a `SummarizedExperiment`.

        Args:
            args (Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]]): indices to slice. tuple can
                contains slices along dimensions.

        Raises:
            ValueError: Too many or few slices.

        Returns:
            SummarizedExperiment: sliced `SummarizedExperiment` object.
        """
        new_rows, new_cols, new_assays = self._slice(args)
        return SummarizedExperiment(
            assays=new_assays,
            rowData=new_rows,
            colData=new_cols,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        pattern = (
            f"Class SummarizedExperiment with {self.shape[0]} features and {self.shape[1]} samples \n"
            f"  assays: {list(self._assays.keys())} \n"
            f"  features: {self._rows.columns if self._rows is not None else None} \n"
            f"  sample data: {self._cols.columns if self._cols is not None else None}"
        )
        return pattern
