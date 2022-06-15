from typing import Any, Dict, Union
from scipy import sparse as sp
import numpy as np
import pandas as pd
from genomicranges import GenomicRanges
from .SummarizedExperiment import SummarizedExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class RangeSummarizedExperiment(SummarizedExperiment):
    """SummarizedExperiment class to represent genomic experiment data,
    features, sample data and any other metadata
    """

    def __init__(
        self,
        assays: Dict[str, Union[np.ndarray, sp.spmatrix]],
        rows: GenomicRanges = None,
        cols: pd.DataFrame = None,
        metadata: Any = None,
    ) -> None:
        """Initialize an instance of `RangeSummarizedExperiment`

        Args:
            assays (Dict[str, Union[np.ndarray, sp.spmatrix]]): list of matrices,
                represented as dense (numpy) or sparse (scipy) matrices
            rows (pd.DataFrame): features. Defaults to None.
            cols (pd.DataFrame): sample metadata. Defaults to None.
            metadata (Any, optional): experiment metadata describing the
                methods. Defaults to None.
        """
        super().__init__(assays, rows, cols, metadata=metadata)

    def rowRanges(self) -> GenomicRanges:
        """Accessor to retrieve features

        Returns:
            GenomicRanges: returns features in the container
        """
        return self.rows

    def __getitem__(self, args: tuple) -> "RangeSummarizedExperiment":
        """Subset a SummarizedExperiment

        Args:
            args (tuple): indices to slice. tuple can
                contains slices along dimensions

        Returns:
            RangeSummarizedExperiment: new SummarizedExperiment object
        """
        rowIndices = args[0]
        colIndices = None
        if len(args) > 1:
            colIndices = args[1]

        new_rows = None
        new_cols = None
        new_assays = None

        if rowIndices is not None:
            new_rows = self.rows[rowIndices]

        if colIndices is not None:
            new_cols = self.cols.iloc[colIndices]

        new_assays = self.subsetAssays(
            rowIndices=rowIndices, colIndices=colIndices
        )

        return RangeSummarizedExperiment(
            new_assays, new_rows, new_cols, self.metadata
        )

    def subsetByOverlaps(
        self, query: GenomicRanges
    ) -> "RangeSummarizedExperiment":
        """Subset a RangeSummarizedExperiment by feature overlaps

        Args:
            query (GenomicRanges): query genomic intervals

        Raises:
            NotImplementedError: Currently not implemented

        Returns:
            RangeSummarizedExperiment: new SummarizedExperiment object
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """string representation

        Returns:
            str: description of the class
        """
        return (
            "Class: RangeSummarizedExperiment\n"
            f"\tshape: {self.shape}\n"
            f"\tcontains assays: {self._assays.keys()}\n"
            f"\tsample metadata: {self.cols.columns}\n"
            f"\tfeatures: {self.rows}\n"
        )
