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
    features, sample data and any other metadata
    """

    def __init__(
        self,
        assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
        rowData: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        colData: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        metadata: MutableMapping = None,
    ) -> None:
        super().__init__(assays, rowData, colData, metadata)

    @property
    def rowData(self) -> Optional[Union[pd.DataFrame, BiocFrame]]:
        """Access features.

        Returns:
            Optional[Union[pd.DataFrame, BiocFrame]]: returns features.
        """
        return self._rows

    @rowData.setter
    def rowData(self, rows: Optional[Union[pd.DataFrame, BiocFrame]]) -> None:
        """Set features.

        Args:
            rows (Optional[Union[pd.DataFrame, BiocFrame]]): features to update
        """
        self._rows = rows
        self._validate()

    def __getitem__(
        self,
        args: Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]],
    ) -> "SummarizedExperiment":
        """Subset a `SummarizedExperiment`.

        Args:
            args (Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]]): indices to slice. tuple can
                contains slices along dimensions

        Raises:
            ValueError: Too many slices

        Returns:
            SummarizedExperiment: new sliced SummarizedExperiment object
        """
        new_rows, new_cols, new_assays = self._slice(args)
        return SummarizedExperiment(
            assays=new_assays,
            rowData=new_rows,
            colData=new_cols,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        pattern = """
        Class SummarizedExperiment with {} features and {} samples
          assays: {}
          features: {}
          sample data: {}
        """
        return pattern.format(
            self.shape[0],
            self.shape[1],
            list(self._assays.keys()),
            self.cols.columns if self.cols is not None else None,
            self.rows.columns if self.rows is not None else None,
        )
