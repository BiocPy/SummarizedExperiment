from typing import Union, MutableMapping, Sequence, Tuple, Optional
from biocframe import BiocFrame
from genomicranges import GenomicRanges
from scipy import sparse as sp
import numpy as np
import pandas as pd
from collections import OrderedDict
import anndata

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class BaseSE:
    """Base class for Summarized Experiment.
    
    Represents genomic experiment data (`assays`), features (`rowdata`), 
    sample data (`coldata`) and any other metadata.
    """

    _shape = None

    def __init__(
        self,
        assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
        rows: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        cols: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize an instance of `BaseSE`.

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

        if assays is None or not isinstance(assays, dict) or len(assays.keys()) == 0:
            raise Exception(
                f"{assays} must be a dictionary and contain "
                "atleast one matrix (either sparse or dense)"
            )

        self._validate_assays(assays)
        self._assays = assays

        rows = rows if rows is not None else BiocFrame({}, numberOfRows=self._shape[0])
        self._validate_rows(rows)
        self._rows = rows

        cols = cols if cols is not None else BiocFrame({}, numberOfRows=self._shape[1])
        self._validate_cols(cols)
        self._cols = (
            cols if cols is not None else BiocFrame({}, numberOfRows=self._shape[1])
        )

        self._metadata = metadata

    def _validate(self):
        """Internal wrapper method to validate the object.
        """
        # validate assays to make sure they are have same dimensions
        self._validate_assays(self._assays)
        self._validate_rows(self._rows)
        self._validate_cols(self._cols)

    def _validate_assays(
        self, assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]]
    ):
        """Internal method to validate experiment data (assays).

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): experiment
                data.

        Raises:
            ValueError: when assays contain more than 2 dimensions.
            ValueError: if all assays do not have the same dimensions.
        """
        for asy, mat in assays.items():
            if len(mat.shape) > 2:
                raise ValueError(
                    "only 2-dimensional matrices are accepted, "
                    f"provided {len(mat.shape)} dimensions for assay {asy}"
                )

            if self._shape is None:
                self._shape = mat.shape
                continue

            if mat.shape != self._shape:
                raise ValueError(
                    f"Assay: {asy} must be of shape {self._shape}"
                    f" but provided {mat.shape}"
                )

    def _validate_rows(self, rows: Optional[Union[pd.DataFrame, BiocFrame]]):
        """Internal method to validate feature information (rowdata).

        Args:
            rows (Optional[Union[pd.DataFrame, BiocFrame]]): feature information 
                (rowdata).

        Raises:
            ValueError: when number of rows does not match between rows & assays.
            TypeError: when rows is neither a pandas dataframe not Biocframe object.
        """
        if not (isinstance(rows, pd.DataFrame) or isinstance(rows, BiocFrame)):
            raise TypeError(
                "rowData must be either a pandas `DataFrame` or a `BiocFrame`"
                f" object, provided {type(rows)}"
            )

        if rows.shape[0] != self._shape[0]:
            raise ValueError(
                f"Features and assays do not match. must be {self._shape[0]}"
                f" but provided {rows.shape[0]}"
            )

    def _validate_cols(self, cols: Optional[Union[pd.DataFrame, BiocFrame]]):
        """Internal method to validate sample information (coldata).

        Args:
            cols (Optional[Union[pd.DataFrame, BiocFrame]]): sample information 
                (coldata).

        Raises:
            ValueError: when number of samples do not match between cols & assays.
            TypeError: when cols is neither a pandas dataframe not Biocframe object.
        """
        if not (isinstance(cols, pd.DataFrame) or isinstance(cols, BiocFrame)):
            raise TypeError(
                "colData must be either a pandas `DataFrame` or a `BiocFrame`"
                f" object, provided {type(cols)}"
            )

        if cols.shape[0] != self._shape[1]:
            raise ValueError(
                f"Sample data and assays do not match. must be {self._shape[1]}"
                f" but provided {cols.shape[0]}"
            )

    @property
    def assays(self) -> MutableMapping[str, Union[np.ndarray, sp.spmatrix]]:
        """Get assays.

        Returns:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix]]: a dictionary with  
            experiments names as keys and matrix data as values.
        """
        return self._assays

    @assays.setter
    def assays(
        self, assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]]
    ) -> None:
        """Set new experiment data (assays).

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): new assays.
        """
        self._validate_assays(assays)
        self._assays = assays

    @property
    def colData(self) -> Union[pd.DataFrame, BiocFrame]:
        """Get sample data.

        Returns:
            Union[pd.DataFrame, BiocFrame]: Sample information.
        """
        return self._cols

    @colData.setter
    def colData(self, cols: Union[pd.DataFrame, BiocFrame]) -> None:
        """Set sample data.

        Args:
            cols (Union[pd.DataFrame, BiocFrame]): sample data to update.
        """
        cols = cols if cols is not None else BiocFrame({}, numberOfRows=self._shape[1])

        self._validate_cols(cols)
        self._cols = cols

    @property
    def metadata(self) -> Optional[MutableMapping]:
        """Get metadata.

        Returns:
            Optional[MutableMapping]: Metadata object, usually a dictionary.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[MutableMapping]):
        """Set metadata.

        Args:
            metadata (Optional[MutableMapping]): new metadata object.
        """
        self._metadata = metadata

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the experiment, (number of features and number of samples).

        Returns:
            Tuple[int, int]: A tuple with number of features and number of samples.
        """
        return self._shape

    @property
    def dims(self) -> Tuple[int, int]:
        """Dimensions of the experiment, (number of features and number of samples).

        Note: same as shape.

        Returns:
            Tuple[int, int]: A tuple with number of features and number of samples.
        """
        return self.shape

    @property
    def assayNames(self) -> Sequence[str]:
        """Get assay names.

        Returns:
            Sequence[str]: list of assay names.
        """
        return list(self._assays.keys())

    @assayNames.setter
    def assayNames(self, names: Sequence[str]):
        """Replace all assay names.

        Args:
            names (Sequence[str]): new names.

        Raises:
            ValueError: if enough names are not provided.
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
        pattern = (
            f"Class BaseSE with {self.shape[0]} features and {self.shape[1]} samples \n"
            f"  assays: {list(self._assays.keys())} \n"
            f"  features: {self._rows.columns if self._rows is not None else None} \n"
            f"  sample data: {self._cols.columns if self._cols is not None else None}"
        )
        return pattern

    def assay(self, name: str) -> Union[np.ndarray, sp.spmatrix]:
        """Convenience function to access an assay by name.

        Args:
            name (str): name of the assay.

        Raises:
            ValueError: if assay name does not exist.

        Returns:
            Union[np.ndarray, sp.spmatrix]: experiment data.
        """
        if name not in self._assays:
            raise ValueError(f"Assay {name} does not exist")

        return self._assays[name]

    def subsetAssays(
        self,
        rowIndices: Optional[Union[Sequence[int], slice]] = None,
        colIndices: Optional[Union[Sequence[int], slice]] = None,
    ) -> MutableMapping[str, Union[np.ndarray, sp.spmatrix]]:
        """Subset all assays to a specified set of {rows, cols or both} slices.

        Args:
            rowIndices (Union[Sequence[int], slice], optional): row indices to subset. 
                Defaults to None.
            colIndices (Union[Sequence[int], slice], optional): col indices to subset. 
                Defaults to None.

        Raises:
            ValueError: if `rowIndices` and `colIndices` are both None.

        Returns:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix]]: experiment data
            for only the specified slices.
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
        """Internal method to slice `SE` by index.

        Args:
            args (Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]]): 
                indices to slice. tuple contains slices along dimensions (rows, cols).

        Raises:
            ValueError: Too many or too few slices provided.

        Returns:
            Tuple[Union[pd.DataFrame, BiocFrame], Union[pd.DataFrame, BiocFrame], MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]: 
            sliced row, cols and assays.
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
        """Get row/feature index.

        Returns:
            Sequence[str]: list of row index names.
        """
        if isinstance(self._rows, pd.DataFrame):
            return self._rows.index.tolist()
        else:
            return self._rows.rowNames

    @rownames.setter
    def rownames(self, names: Sequence[str]):
        """Set row/feature names for the experiment.

        Args:
            names (Sequence[str]): new feature names.

        Raises:
            ValueError: provided incorrect number of names.
        """
        if len(names) != self.shape[0]:
            raise ValueError(
                f"names must be of length {self.shape[0]}, provided {len(names)}"
            )

        if isinstance(self._rows, pd.DataFrame):
            self._rows.index = names
        else:
            self._rows.rowNames = names

    @property
    def colnames(self) -> Sequence[str]:
        """Get column/sample names.

        Returns:
            Sequence[str]: list of sample names.
        """
        if isinstance(self._cols, pd.DataFrame):
            return self._cols.index.tolist()
        else:
            return self._cols.rowNames

    @colnames.setter
    def colnames(self, names: Sequence[str]):
        """Set column/sample names for the experiment.

        Args:
            names (Sequence[str]): new samples names.
        """
        if len(names) != self.shape[1]:
            raise ValueError(
                f"names must be of length {self.shape[1]}, provided {len(names)}"
            )

        if isinstance(self._cols, pd.DataFrame):
            self._cols.index = names
        else:
            self._cols.rowNames = names

    def toAnnData(self,) -> anndata.AnnData:
        """Transform `SummarizedExperiment` to `AnnData` representation. 

        Returns:
            anndata.AnnData: returns an `AnnData` representation of SE.
        """

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()

        trows = self._rows
        if isinstance(self._rows, GenomicRanges):
            trows = self._rows.toPandas()

        obj = anndata.AnnData(
            obs=self._cols, var=trows, uns=self.metadata, layers=layers,
        )

        return obj
