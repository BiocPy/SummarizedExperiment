from typing import Union, MutableMapping, Sequence, Tuple, Optional
from biocframe import BiocFrame
from scipy import sparse as sp
import numpy as np
import pandas as pd
from collections import OrderedDict

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class BaseSE:
    """BaseSE class to represent genomic experiment data,
    features, sample data and any other metadata
    """

    def __init__(
        self,
        assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
        rows: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        cols: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize an instance of `BaseSE`.

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): dictionary of matrices,
                with assay names as keys and matrices represented as dense (numpy) or sparse (scipy) matrices.
                All matrices across assays must have the same dimensions (number of rows, number of columns).
            rows (Union[pd.DataFrame, BiocFrame], optional): features, must be the same length as rows of the matrices in assays. Defaults to None.
            cols (Union[pd.DataFrame, BiocFrame], optional): sample data, must be the same length as the columns of the matrices in assays. Defaults to None.
            metadata (MutableMapping, optional): experiment metadata describing the methods. Defaults to None.
        """

        if assays is None or not isinstance(assays, dict) or len(assays.keys()) == 0:
            raise Exception(
                f"{assays} must be a dictionary and contain atleast one matrix (either sparse or dense)"
            )

        self._rows = rows
        self._assays = assays
        self._cols = cols
        self._metadata = metadata

        self._validate()

    def _validate_types(self):
        """internal method to validate types
        """

        if self._rows is not None:
            if not (
                isinstance(self._rows, pd.DataFrame)
                or isinstance(self._rows, BiocFrame)
            ):
                raise TypeError(
                    "rowData must be either a pandas `DataFrame` or a `BiocFrame` object"
                )

        if self._cols is not None:
            if not (
                isinstance(self._cols, pd.DataFrame)
                or isinstance(self._cols, BiocFrame)
            ):
                raise TypeError(
                    "colData must be either a pandas `DataFrame` or a `BiocFrame` object"
                )

    def _validate(self):
        """Internal method to validate the object

        Raises:
            ValueError: when provided object does not contain columns of same length
        """

        self._validate_types()

        # validate assays to make sure they are all same dimensions
        base_dims = None
        for asy, mat in self._assays.items():
            if len(mat.shape) > 2:
                raise ValueError(
                    f"only 2-dimensional matrices are accepted, provided {len(mat.shape)} dimensions for assay {asy}"
                )

            if base_dims is None:
                base_dims = mat.shape
                continue

            if mat.shape != base_dims:
                raise ValueError(
                    f"Assay: {asy} must be of shape {base_dims} but provided {mat.shape}"
                )

        # check feature length
        if self._rows is not None:
            if self._rows.shape[0] != base_dims[0]:
                raise ValueError(
                    f"Features and assays do not match. must be {base_dims[0]} but provided {self._rows.shape[0]}"
                )

        # check sample length
        if self._cols is not None:
            if self._cols.shape[0] != base_dims[1]:
                raise ValueError(
                    f"Sample data and assays do not match. must be {base_dims[1]} but provided {self._cols.shape[0]}"
                )

        self._shape = base_dims

    @property
    def assays(self) -> MutableMapping[str, Union[np.ndarray, sp.spmatrix]]:
        """Get assays.

        Returns:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix]]: a dictionary of experiments.
        """
        return self._assays

    @assays.setter
    def assays(
        self, assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]]
    ) -> None:
        """Set assays.

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): sets the assays
        """
        self._assays = assays
        self._validate()

    @property
    def colData(self) -> Optional[Union[pd.DataFrame, BiocFrame]]:
        """Get sample data.

        Returns:
            Optional[Union[pd.DataFrame, BiocFrame]]: returns sample data.
        """
        return self._cols

    @colData.setter
    def colData(self, cols: Optional[Union[pd.DataFrame, BiocFrame]]) -> None:
        """Set sample data.

        Args:
            cols (Optional[Union[pd.DataFrame, BiocFrame]]): sample data to update.
        """
        self._cols = cols
        self._validate()

    @property
    def metadata(self) -> Optional[MutableMapping]:
        """Get metadata.

        Returns:
            Optional[MutableMapping]: metadata object, usually a dictionary.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[MutableMapping]):
        """Set metadata.

        Args:
            metadata (Optional[MutableMapping]): new metadata object
        """
        self._metadata = metadata

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the object, number of features and number of samples.

        Returns:
            Tuple[int, int]: A tuple of number of features and number of samples
        """
        a_mat = self._assays[list(self._assays.keys())[0]]
        return a_mat.shape

    @property
    def dims(self) -> Tuple[int, int]:
        """Dimensions of the object, number of features and number of samples.

        Returns:
            Tuple[int, int]: A tuple of number of features and number of samples
        """
        return self.shape

    @property
    def assayNames(self) -> Sequence[str]:
        """Get assay names.

        Returns:
            Sequence[str]: list of assay names
        """
        return list(self._assays.keys())

    @assayNames.setter
    def assayNames(self, names: Sequence[str]):
        """Set assay names.

        Args:
            names (Sequence[str]): new names

        Raises:
            ValueError: if enough names are not provided
        """
        current_names = list(self._assays.keys())
        if len(names) != len(current_names):
            raise ValueError(
                f"names must be of length {len(current_names)}, provided {len(names)}"
            )

        new_assays = OrderedDict()
        for idx in range(len(names)):
            new_assays[names[idx]] = self._assays.pop(current_names[idx])

        self._assays = new_assays

    def __str__(self) -> str:
        pattern = """
        Class BaseSE with {} features and {} samples
          assays: {}
          features: {}
          sample data: {}
        """
        return pattern.format(
            self.shape[0],
            self.shape[1],
            list(self._assays.keys()),
            self._rows.columns if self._rows is not None else None,
            self._cols.columns if self._cols is not None else None,
        )

    def assay(self, name: str) -> Union[np.ndarray, sp.spmatrix]:
        """Convenience function to access an assay by name

        Args:
            name (str): name of the assay

        Raises:
            ValueError: if assay name does not exist

        Returns:
            Union[np.ndarray, sp.spmatrix]: The experiment data
        """
        if name not in self._assays:
            raise ValueError(f"Assay {name} does not exist")

        return self._assays[name]

    def subsetAssays(
        self,
        rowIndices: Union[Sequence[int], slice] = None,
        colIndices: Union[Sequence[int], slice] = None,
    ) -> MutableMapping[str, Union[np.ndarray, sp.spmatrix]]:
        """Subset all assays to a specified rows or cols or both

        Args:
            rowIndices (Union[Sequence[int], slice], optional): row indices to subset. Defaults to None.
            colIndices (Union[Sequence[int], slice], optional): col indices to subset. Defaults to None.

        Raises:
            ValueError: if rowIndices and colIndices are both None.

        Returns:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix]]: assays with subset matrices
        """

        if rowIndices is None and colIndices is None:
            raise ValueError("either `rowIndices` and `colIndices` must be provided")

        new_assays = OrderedDict()
        for asy, mat in self._assays.items():
            if rowIndices is not None:
                mat = mat[rowIndices, :]

            if colIndices is not None:
                mat = mat[:, colIndices]

            new_assays[asy] = mat

        return new_assays

    def _slice(
        self,
        args: Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]],
    ) -> Tuple[
        Union[pd.DataFrame, BiocFrame],
        Union[pd.DataFrame, BiocFrame],
        MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
    ]:
        """Internal method to slice `SE` by index

        Args:
            args (Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]]): indices to slice. tuple can
                contains slices along dimensions

        Raises:
            ValueError: Too many slices

        Returns:
             Tuple[Union[pd.DataFrame, BiocFrame], Union[pd.DataFrame, BiocFrame], MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]: sliced row, cols and assays.
        """

        if len(args) == 0:
            raise ValueError("Arguments must contain one slice")

        rowIndices = args[0]
        colIndices = None

        if len(args) > 1:
            colIndices = args[1]
        elif len(args) > 2:
            raise ValueError("contains too many slices")

        new_rows = None
        new_cols = None
        new_assays = None

        if rowIndices is not None and self._rows is not None:
            if isinstance(self._rows, pd.DataFrame):
                new_rows = self._rows.iloc[rowIndices]
            else:
                new_rows = self._rows[rowIndices, :]

        if colIndices is not None and self._cols is not None:
            if isinstance(self._cols, pd.DataFrame):
                new_cols = self._cols.iloc[colIndices]
            else:
                new_cols = self._cols[colIndices, :]

        new_assays = self.subsetAssays(rowIndices=rowIndices, colIndices=colIndices)

        return (new_rows, new_cols, new_assays)

    @property
    def rownames(self) -> Sequence[str]:
        """Get row/feature index

        Returns:
            Sequence[str]: list of row index names
        """
        if isinstance(self._rows, pd.DataFrame):
            return self._rows.index.tolist()
        else:
            return self._rows.rowNames

    @property
    def colnames(self) -> Sequence[str]:
        """Get column/sample names

        Returns:
            Sequence[str]: list of column names
        """
        if isinstance(self._cols, pd.DataFrame):
            return self._cols.index.tolist()
        else:
            return self._cols.rowNames