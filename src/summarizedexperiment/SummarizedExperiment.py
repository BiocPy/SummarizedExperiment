from typing import Any, Dict, List, Union
from scipy import sparse as sp
import numpy as np
import pandas as pd
import logging

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment:
    """SummarizedExperiment class to represent genomic experiment data,
    features, sample data and any other metadata
    """

    def __init__(
        self,
        assays: Dict[str, Union[np.ndarray, sp.spmatrix]],
        rows: pd.DataFrame = None,
        cols: pd.DataFrame = None,
        metadata: Any = None,
    ) -> None:
        """Initialize an instance of `SummarizedExperiment`

        Args:
            assays (Dict[str, Union[np.ndarray, sp.spmatrix]]): list of matrices,
                represented as dense (numpy) or sparse (scipy) matrices
            rows (pd.DataFrame): features. Defaults to None.
            cols (pd.DataFrame): sample metadata. Defaults to None.
            metadata (Any, optional): experiment metadata describing the
                methods. Defaults to None.
        """

        if (
            assays is None
            or not isinstance(assays, dict)
            or len(assays.keys()) == 0
        ):
            raise Exception(
                f"{assays} must be a dictionary and contain atleast a single numpy/scipy matrix"
            )

        self.rows = rows
        self._assays = assays
        self.cols = cols
        self.metadata = metadata

    def assays(self) -> Dict[str, Union[np.ndarray, sp.spmatrix]]:
        """Accessor to retrieve experimental data. This is a
        dictionary containing more than one assay and can be
        directly accessed using subset (`[]`) operator.

        Returns:
            Dict[str, Union[np.ndarray, sp.spmatrix]]: Dictionary of experiments
        """
        return self._assays

    def rowData(self) -> pd.DataFrame:
        """Accessor to retrieve features

        Returns:
            pd.DataFrame: returns features in the container
        """
        return self.rows

    def colData(self) -> pd.DataFrame:
        """Accessor to retrieve sample metadata

        Returns:
            pd.DataFrame: returns features in the container
        """
        return self.cols

    def subsetAssays(
        self, rowIndices: List[int] = None, colIndices: List[int] = None
    ) -> Dict[str, Union[np.ndarray, sp.spmatrix]]:
        """Subset all assays to a specific row or col indices

        Args:
            rowIndices (List[int], optional): row indices to subset. Defaults to None.
            colIndices (List[int], optional): col indices to subset. Defaults to None.

        Returns:
            Dict[str, Union[np.ndarray, sp.spmatrix]]: assays with subset matrices
        """

        new_assays = {}
        for key in self._assays:
            mat = self._assays[key]
            if rowIndices is not None:
                mat = mat[rowIndices, :]

            if colIndices is not None:
                mat = mat[:, colIndices]

            new_assays[key] = mat

        return new_assays

    def __getitem__(self, args: tuple) -> "SummarizedExperiment":
        """Subset a SummarizedExperiment

        Args:
            args (tuple): indices to slice. tuple can
                contains slices along dimensions

        Raises:
            Exception: Too many slices

        Returns:
            SummarizedExperiment: new SummarizedExperiment object
        """
        rowIndices = args[0]
        colIndices = None

        if len(args) > 1:
            colIndices = args[1]
        elif len(args) > 2:
            logging.error(
                f"too many slices, args length must be 2 but provided {len(args)} "
            )
            raise Exception("contains too many slices")

        new_rows = None
        new_cols = None
        new_assays = None

        if rowIndices is not None and self.rows is not None:
            new_rows = self.rows.iloc[rowIndices]

        if colIndices is not None and self.cols is not None:
            new_cols = self.cols.iloc[colIndices]

        new_assays = self.subsetAssays(
            rowIndices=rowIndices, colIndices=colIndices
        )

        return SummarizedExperiment(
            new_assays, new_rows, new_cols, self.metadata
        )

    def metadata(self) -> Any:
        """Access Experiment metadata

        Returns:
            Any: metadata (could be anything)
        """
        return self.metadata

    @property
    def shape(self) -> tuple:
        """Get shape of the object

        Returns:
            tuple: dimensions of the experiment; (rows, cols)
        """
        a_mat = self._assays[self._assays.keys()[0]]
        return a_mat.shape

    def __str__(self) -> str:
        """string representation

        Returns:
            str: description of the class
        """
        return (
            "Class: SummarizedExperiment\n"
            f"\tshape: {self.shape}\n"
            f"\tcontains assays: {self._assays.keys()}\n"
            f"\tsample metadata: {self.cols.columns if self.cols is not None else None}\n"
            f"\tfeatures: {self.rows.columns if self.rows is not None else None}\n"
        )
