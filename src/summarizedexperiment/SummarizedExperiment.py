"""The SummarizedExperiment class."""

from typing import Any, Dict, MutableMapping, Optional, Sequence, Tuple, Union

from anndata import AnnData  # type: ignore
from biocframe import BiocFrame  # type: ignore
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import spmatrix  # type: ignore

from .BaseSE import BaseSE

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """SummarizedExperiment class.

    Contains experiment, feature, sample, and meta- data.
    """

    def __init__(
        self,
        assays: MutableMapping[str, Union[NDArray[Any], spmatrix]],
        rowData: Union[DataFrame, BiocFrame, None] = None,
        colData: Union[DataFrame, BiocFrame, None] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        """Initialize an instance of `BaseSE`.

        Args:
            assays:
                A `dict` of matrices, with assay names as keys and matrices represented
                as dense (numpy) or sparse (scipy) matrices. All matrices across assays
                must have the same dimensions (number of rows, number of columns).
            rowData:
                Features, must be the same length as rows of the matrices in assays.
                Defaults to `None`.
            colData:
                Sample data, must be the same length as the columns of the matrices in
                assays. Defaults to `None`.
            metadata:
                Experiment metadata describing the methods. Defaults to `None`.

        Raises:
            ValueError: when assays is empty.
            ValueError: when assays are not the same shape.
            ValueError: when rows are not the same length as the number of rows in
                assays.
            ValueError: when cols are not the same length as the number of cols in
                assays.
        """
        self._shape: Tuple[int, int] = (0, 0)
        self._validate_assays(assays)
        self._assays = assays

        rowData = (
            BiocFrame(numberOfRows=self._shape[0])
            if rowData is None
            else rowData
        )
        self._validate_rows(rowData)
        self._rows = rowData

        colData = (
            BiocFrame(numberOfRows=self._shape[1])
            if colData is None
            else colData
        )
        self._validate_cols(colData)
        self._cols = colData

        self._metadata = metadata

    def _validate_rows(self, rowData: Union[DataFrame, BiocFrame]) -> None:
        """Validate rows.

        Args:
            rowData: Row data to validate.

        Raises:
            ValueError: when rows are not the same length as the number of rows in
                assays.
        """
        if not isinstance(rowData, (DataFrame, BiocFrame)):  # type: ignore
            raise ValueError(
                f"rowData be either a pandas DataFrame or a BiocFrame, "
                f"provided {type(rowData)}"
            )

        if rowData.shape[0] != self._shape[0]:
            raise ValueError(
                f"Number of rows in rows ({rowData.shape[0]}) must match the number of "
                f"rows in assays ({self._shape[0]})"
            )

    @property
    def rowData(self) -> Union[DataFrame, BiocFrame]:
        """Get row/feature data.

        Args:
            rows: New feature data.

        Returns:
            Current feature data.
        """
        return self._rows

    @rowData.setter
    def rowData(self, rows: Union[DataFrame, BiocFrame]) -> None:
        self._validate_rows(rows)
        self._rows = rows

    @property
    def rownames(self) -> Sequence[str]:
        """Get row/feature names.

        Returns:
            A list of row index names.
        """
        if isinstance(self._rows, DataFrame):
            return self._rows.index.tolist()  # type: ignore
        else:
            return self._rows.rowNames

    def _slice(
        self, args: Tuple[Union[Sequence[int], slice], ...]
    ) -> Tuple[
        MutableMapping[str, Union[NDArray[Any], spmatrix]],
        Union[DataFrame, BiocFrame],
        Union[DataFrame, BiocFrame],
    ]:
        """Internal method to slice `RSE` by index.

        Args:
            args:
                Indices to slice, `tuple` can contain slices along dimensions, max 2
                dimensions accepted.

        Raises:
            ValueError: Too many slices

        Returns:
            Sliced assays. Sliced rows. Sliced cols.
        """
        if len(args) == 0:
            raise ValueError("Arguments must contain at least one slice.")

        rowIndices = args[0]
        colIndices = None

        if len(args) > 1:
            colIndices = args[1]
        elif len(args) > 2:
            raise ValueError("Parameter 'args' contains too many slices.")

        if isinstance(self._rows, DataFrame):
            new_rows: Union[DataFrame, BiocFrame] = self._rows.iloc[rowIndices]  # type: ignore
        else:
            new_rows = self._rows[rowIndices, :]

        new_cols: Union[DataFrame, BiocFrame] = self._cols
        if colIndices is not None:
            if isinstance(self._cols, DataFrame):
                new_cols = self._cols.iloc[colIndices]  # type: ignore
            else:
                new_cols = self._cols[colIndices, :]

        new_assays = self.subsetAssays(
            rowIndices=rowIndices, colIndices=colIndices
        )

        return (new_assays, new_rows, new_cols)

    def __str__(self) -> str:
        """The string representation of the experiment."""
        pattern = (
            "Class: SummarizedExperiment with {} features and {} samples\n"
            "   assays: {}\n"
            "   features: {}\n"
            "   sample data: {}\n"
        )

        return pattern.format(
            self.shape[0],
            self.shape[1],
            list(self._assays.keys()),
            self._rows.columns,
            self._cols.columns,
        )

    def __getitem__(
        self, args: Tuple[Union[Sequence[int], slice], ...]
    ) -> "SummarizedExperiment":
        """Subset a `SummarizedExperiment`.

        Args:
            args:
                Indices to slice, `tuple` can contain slices along dimensions, max 2
                dimensions accepted.

        Raises:
            ValueError: Too many slices

        Returns:
            A new sliced `RangeSummarizedExperiment` object.
        """
        return SummarizedExperiment(*self._slice(args), self.metadata)

    def toAnnData(self) -> AnnData:
        """Transform `SummarizedExperiment` object to `AnnData`.

        Returns:
            An `AnnData` representation of the RSE.
        """
        layers: Dict[str, Any] = {}
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()  # type: ignore

        return AnnData(
            obs=self.colData
            if isinstance(self.colData, DataFrame)
            else self.colData.toPandas(),
            var=self.rowData
            if isinstance(self.rowData, DataFrame)
            else self.rowData.toPandas(),
            uns=self.metadata,
            layers=layers,
        )
