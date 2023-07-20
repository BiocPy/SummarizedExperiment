import warnings
from collections import OrderedDict
from functools import reduce
from typing import MutableMapping, Optional, Sequence, Tuple, Union, Literal

import anndata
import numpy as np
import pandas as pd
from pandas import DataFrame
from biocframe import BiocFrame
from filebackedarray import H5BackedDenseData, H5BackedSparseData
from genomicranges import GenomicRanges
from scipy import sparse as sp

from .dispatchers.colnames import get_colnames, set_colnames
from .dispatchers.rownames import get_rownames, set_rownames

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class BaseSE:
    """Base class for Summarized Experiment.

    Represents genomic experiment data (`assays`), features (`rowdata`),
    sample data (`coldata`) and any other metadata.

    Args:
        assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]]): dictionary
            of matrices, with assay names as keys and matrices represented as dense
            (numpy) or sparse (scipy) matrices. All matrices across assays must
            have the same dimensions (number of rows, number of columns).
        rowData (Union[pd.DataFrame, BiocFrame], optional): features, must be the same length as
            rows of the matrices in assays. Defaults to None.
        colData (Union[pd.DataFrame, BiocFrame], optional): sample data, must be
            the same length as rows of the matrices in assays. Defaults to None.
        metadata (MutableMapping, optional): experiment metadata describing the
            methods. Defaults to None.
    """

    # TODO: should be an instance attribute
    _shape = None

    def __init__(
        self,
        assays: MutableMapping[
            str, Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]
        ],
        rows: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        cols: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize an instance of `BaseSE`."""

        if assays is None or not isinstance(assays, dict) or len(assays.keys()) == 0:
            raise Exception(
                f"{assays} must be a dictionary and contain "
                "atleast one matrix (either sparse or dense)"
            )

        self._validate_assays(assays)
        self._assays = assays

        # should have _shape by now
        if self._shape is None:
            raise TypeError(
                "This should not happen; assays is not consistent. "
                "Report this issue with a reproducible example!"
            )

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
        """Internal wrapper method to validate the object."""
        # validate assays to make sure they are have same dimensions
        self._validate_assays(self._assays)
        self._validate_rows(self._rows)
        self._validate_cols(self._cols)

    def _validate_assays(
        self,
        assays: MutableMapping[
            str, Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]
        ],
    ):
        """Internal method to validate experiment data (assays).

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]]): experiment
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
    def assays(
        self,
    ) -> MutableMapping[
        str, Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]
    ]:
        """Get assays.

        Returns:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]]: a dictionary with
            experiments names as keys and matrix data as values.
        """
        return self._assays

    @assays.setter
    def assays(
        self,
        assays: MutableMapping[
            str, Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]
        ],
    ) -> None:
        """Set new experiment data (assays).

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]]): new assays.
        """
        self._validate_assays(assays)
        self._assays = assays

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
        rows = rows if rows is not None else BiocFrame({}, numberOfRows=self.shape[0])
        self._validate_rows(rows)
        self._rows = rows

    @property
    def colData(self) -> Optional[Union[pd.DataFrame, BiocFrame]]:
        """Get sample data.

        Returns:
            (Union[pd.DataFrame, BiocFrame], optional): Sample information.
        """
        return self._cols

    @colData.setter
    def colData(self, cols: Optional[Union[pd.DataFrame, BiocFrame]]) -> None:
        """Set sample data.

        Args:
            cols (Union[pd.DataFrame, BiocFrame], optional): sample data to update.
        """
        cols = cols if cols is not None else BiocFrame({}, numberOfRows=self.shape[1])

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
        """Get shape of the experiment.

        Returns:
            Tuple[int, int]: A tuple with (number of features, number of samples).
        """
        return self._shape

    @property
    def dims(self) -> Tuple[int, int]:
        """Dimensions of the experiment, similar to shape.

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
        current_names = self.assayNames
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

    def assay(
        self, name: str
    ) -> Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]:
        """Convenience function to access an assay by name.

        Args:
            name (str): name of the assay.

        Raises:
            ValueError: if assay name does not exist.

        Returns:
            Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]: experiment data.
        """
        if name not in self._assays:
            raise ValueError(f"Assay {name} does not exist")

        return self._assays[name]

    def subsetAssays(
        self,
        rowIndices: Optional[Union[Sequence[int], slice]] = None,
        colIndices: Optional[Union[Sequence[int], slice]] = None,
    ) -> MutableMapping[str, Union[np.ndarray, sp.spmatrix]]:
        """Subset all assays to a slice (rows, cols).

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
            warnings.warn("No slice is provided, this returns a copy of all assays!")
            return self.assays.copy()

        new_assays = OrderedDict()
        for asy, mat in self.assays.items():
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
        return get_rownames(self.rowData)

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

        self._rows = set_rownames(self.rowData, names)

    @property
    def colnames(self) -> Sequence[str]:
        """Get column/sample names.

        Returns:
            Sequence[str]: list of sample names.
        """
        return get_colnames(self.colData)

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

        self._cols = set_colnames(self.colData, names)

    def toAnnData(
        self,
    ) -> anndata.AnnData:
        """Transform `SummarizedExperiment` to `AnnData` representation.

        Returns:
            anndata.AnnData: returns an `AnnData` representation of SE.
        """

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            if isinstance(mat, H5BackedDenseData) or isinstance(
                mat, H5BackedSparseData
            ):
                raise ValueError(
                    f"assay {asy} is not supported. Uses a file backed representation."
                    "while this is fine, this is currently not supported because `AnnData` uses"
                    "a transposed representation (cells by features) rather than the "
                    "bioconductor version (features by cells)"
                )

            layers[asy] = mat.transpose()

        trows = self._rows
        if isinstance(self._rows, GenomicRanges):
            trows = self._rows.toPandas()

        obj = anndata.AnnData(
            obs=self._cols,
            var=trows,
            uns=self.metadata,
            layers=layers,
        )

        return obj

    def _validate_row_names(self, rowData: DataFrame) -> bool:
        """Validate that all rowDatas have non-null, non-duplicated row names.

        Args:
            rowData (DataFrame): The rowData to validate.

        Returns:
            bool: True if the rowData has non-null, non-duplicated row names. 
        """
        any_null = rowData.index.isnull().values.any()
        any_duplicated = rowData.index.duplicated().any()
        return (not any_null) and (not any_duplicated)

    def combineCols(
        self,
        *ses: "BaseSE",
        use_names: bool = True
    ) -> "BaseSE":
        """Concatenate multiple `SummarizedExperiment` objects.
        Assume we are working with RNA-Seq experiments for now.

        Args:
            ses ("BaseSE"): `SummarizedExperiment` objects to concatenate.
            use_names (bool):
                If `True`, then each input `SummarizedExperiment` must have non-null, non-duplicated row names.
                The row names of the resultant `SummarizedExperiment` object will be the union of the row
                names across all input objects.
                If `False`, then each input `SummarizedExperiment` object must have the same number of rows.
                The row names of the resultant `SummarizedExperiment` object will simply be the row names of
                the first `SummarizedExperiment`.

        Raises:
            TypeError: if any of the provided objects are not "SummarizedExperiment".
            ValueError: if not all assays have the same dimensions.

        Returns:
            BaseSE: new concatenated `SummarizedExperiment` object.
        """
        all_types = [isinstance(se, BaseSE) for se in ses]
        if not all(all_types):
            raise TypeError("not all provided objects are `SummarizedExperiment` objects")

        all_shapes = [se.shape for se in ses]
        is_all_shapes_same = all_shapes.count(all_shapes[0]) == len(all_shapes)
        if not is_all_shapes_same:
            raise ValueError("not all assays have the same dimensions")
        
        rowDatas = []
        colDatas = []
        for se in ses:
            rowDatas.append(se._rows) # temporary solution
            colDatas.append(se.colData)
        rowDatas.append(self._rows) # temporary solution
        colDatas.append(self.colData)
        
        if use_names:
            is_valid_row_names = all([self._validate_row_names(rowData) for rowData in rowDatas])
            if not is_valid_row_names:
                raise ValueError("at least one input `SummarizedExperiment` has null or duplicated row names")
        else:
            num_rows = [rowData.shape for rowData in rowDatas]
            is_same_num_rows = num_rows.count(num_rows[0]) == len(num_rows)
            if not is_same_num_rows:
                raise ValueError("not all `SummarizedExperiment` objects have the same number of rows. \
                                 Consider setting `use_names=True`")

        new_rowData = reduce(
            lambda left, right: pd.concat([left, right]), rowDatas
        ).sort_index()
        if use_names:
            new_colData = reduce(
                lambda left, right: left.combine_first(right), colDatas
            ).sort_index()
        else:
            new_colData = reduce(
                lambda left, right: pd.concat([left, right]), colDatas
            ).sort_index()

        assays = [se.assays for se in ses]
        assays.append(self.assays)
        assay_names = []
        for assay in assays:
            assay_names.extend(list(assay.keys()))
        unique_assay_names = list(set(assay_names))

        new_assays = {}
        for assay_name in unique_assay_names:
            curr_assays = []
            for se in ses:
                if assay_name not in se.assays:
                    continue
                curr_assay = se.assays[assay_name]
                curr_assays.append(
                    pd.DataFrame(
                        curr_assay, columns=se.colData.index, index=se.rowData.index
                    )
                    if isinstance(curr_assay, np.ndarray)
                    else pd.DataFrame.sparse.from_spmatrix(
                        curr_assay, columns=se.colData.index, index=se.rowData.index
                    )
                )

            merged_assays = reduce(
                lambda left, right: pd.merge(
                    left, right, left_index=True, right_index=True, how=how
                ),
                curr_assays,
            )
            merged_assays = merged_assays.sort_index().sort_values(
                by=merged_assays.columns.tolist()
            )
            merged_assays = sp.csr_matrix(merged_assays.values)
            new_assays[assay_name] = merged_assays

        return SummarizedExperiment(
            assays=new_assays,
            rowData=new_rowData,
            colData=new_colData,
            metadata=metadata
        )
