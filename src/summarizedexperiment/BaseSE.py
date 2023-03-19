"""Base class for SummarizedExperiment types."""

from abc import ABCMeta, abstractmethod
from itertools import groupby
from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from biocframe import BiocFrame  # type: ignore
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import spmatrix  # type: ignore

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class BaseSE(metaclass=ABCMeta):
    """BaseSE metaclass."""

    @abstractmethod
    def _validate_rows(self, rowData: Any) -> None:
        """Validate rows.

        Args:
            rowData: Row data to validate.

        Raises:
            ValueError: when rows are not the same length as the number of rows in
                assays.
        """

    # need a self.rowXXXX property depending on type of data it contains in
    # subclasses

    @property
    @abstractmethod
    def rownames(self) -> Sequence[str]:
        """Get row/feature names.

        Returns:
            A list of row index names.
        """

    @abstractmethod
    def _slice(
        self, args: Tuple[Union[Sequence[int], slice], ...]
    ) -> Tuple[
        MutableMapping[str, Union[NDArray[Any], spmatrix]],
        Any,
        Union[DataFrame, BiocFrame],
    ]:
        """Internal method to slice by index.

        Args:
            args:
                Indices to slice, `tuple` can contain slices along dimensions, max 2
                dimensions accepted.

        Raises:
            ValueError: Too many slices

        Returns:
            Sliced assays. Sliced rows. Sliced cols.
        """

    @abstractmethod
    def __str__(self) -> str:
        """The string representation of the experiment."""

    @abstractmethod
    def __getitem__(
        self, args: Tuple[Union[Sequence[int], slice], ...]
    ) -> "BaseSE":
        """Subset.

        Args:
            args:
                Indices to slice, `tuple` can contain slices along dimensions, max 2
                dimensions accepted.

        Raises:
            ValueError: Too many slices

        Returns:
            A new sliced `RangeSummarizedExperiment` object.
        """

    @abstractmethod
    def toAnnData(self) -> Any:
        """Transform object to `AnnData`.

        Returns:
            An `AnnData` representation of the RSE.
        """

    def _validate_assays(
        self, assays: MutableMapping[str, Union[NDArray[Any], spmatrix]]
    ) -> None:
        """Validate assays.

        Args:
            assays: Assays to validate

        Raises:
            ValueError: When assays are not the same shape.
        """
        if len(assays.keys()) == 0:
            raise ValueError(
                "Assays must contain at least one matrix (either sparse or dense)."
            )

        # validate that all assays are the same shape
        dims: Tuple[List[int], List[int]] = ([], [])
        for name, assay in assays.items():
            if len(assay.shape) != 2:
                raise ValueError(
                    f"Only 2D matrices are accepted, provided {len(assay.shape)} "
                    f"dimensions for assay {name}"
                )

            dims[0].append(assay.shape[0])
            dims[1].append(assay.shape[1])

        group_0 = groupby(dims[0])
        all_same_0 = next(group_0, True) and not next(group_0, False)
        group_1 = groupby(dims[1])
        all_same_1 = next(group_1, True) and not next(group_1, False)

        if not (all_same_0 and all_same_1):
            shapes = {name: assay.shape for name, assay in assays.items()}
            raise ValueError(
                "Assays must be the same shape. Provided assays of different shapes: "
                f"{shapes}"
            )

        self._shape = (dims[0][0], dims[1][0])

    def _validate_cols(self, colData: Union[DataFrame, BiocFrame]) -> None:
        """Validate columns.

        Args:
            colData: Columns to validate

        Raises:
            ValueError: When columns are not the same length.
        """
        if not isinstance(colData, (DataFrame, BiocFrame)):  # type: ignore
            raise ValueError(
                f"colData be either a pandas DataFrame or a BiocFrame, "
                f"provided {type(colData)}"
            )

        if colData.shape[0] != self._shape[1]:
            raise ValueError(
                f"Number of rows in cols ({colData.shape[0]}) must match the number of "
                f"columns in assays ({self._shape[1]})"
            )

    @property
    def assays(self) -> MutableMapping[str, Union[NDArray[Any], spmatrix]]:
        """Get/set the assays.

        Args:
            assays:
                A `dict` of matrices, with assay names as keys and matrices represented
                as dense (numpy) or sparse (scipy) matrices. All matrices across assays
                must have the same dimensions (number of rows, number of columns).

        Returns:
            A `dict` where `key` is the name of the assay and `value` is the assay
                matrix.
        """
        return self._assays

    @assays.setter
    def assays(
        self, assays: MutableMapping[str, Union[NDArray[Any], spmatrix]]
    ) -> None:
        self._validate_assays(assays)
        self._assays = assays

    def assay(self, name: str) -> Union[NDArray[Any], spmatrix]:
        """Convenience function to access an assay by name.

        Args:
            name: Name of the assay

        Raises:
            ValueError: If assay name does not exist

        Returns:
            The assay matrix.
        """
        if name not in self._assays:
            raise ValueError(f"Assay {name} does not exist.")

        return self._assays[name]

    @property
    def colData(self) -> Union[DataFrame, BiocFrame]:
        """Get the column/sample data.

        Args:
            colData: New sample data.

        Returns:
            Current sample data.
        """
        return self._cols

    @colData.setter
    def colData(self, cols: Union[DataFrame, BiocFrame]) -> None:
        self._validate_cols(cols)
        self._cols = cols

    @property
    def metadata(self) -> Optional[MutableMapping[str, Any]]:
        """Get/set the metadata.

        Args:
            metadata: New metadata.

        Returns:
            The current metadata.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[MutableMapping[str, Any]]) -> None:
        self._metadata = metadata

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the experiment data.

        Returns:
            A tuple of the number of features and samples.
        """
        return self._shape

    @property
    def dims(self) -> Tuple[int, int]:
        """The dimensions of the experiment data.

        Same as `shape`.

        Returns:
            A tuple of the number of features and samples.
        """
        return self._shape

    @property
    def assayNames(self) -> List[str]:
        """Get/set the assay names.

        Args:
            names: New assay names.

        Returns:
            A `list` of the assay names.

        Raises:
            ValueError: if the number of names provided does not match the number of
                assays.
        """
        return list(self._assays.keys())

    @assayNames.setter
    def assayNames(self, names: Sequence[str]):
        if len(names) != len(self._assays):
            raise ValueError(
                f"Must provide {len(self._assays)} names, provided {len(names)}"
            )

        self._assays = {
            name: assay for name, assay in zip(names, self._assays.values())
        }

    def subsetAssays(
        self,
        rowIndices: Union[Sequence[int], slice, None] = None,
        colIndices: Union[Sequence[int], slice, None] = None,
    ) -> MutableMapping[str, Union[NDArray[Any], spmatrix]]:
        """Subset all assays to a specified rows or cols or both.

        Args:
            rowIndices: The row indices to subset by. Defaults to `None`.
            colIndices: The col indices to subset by. Defaults to `None`.

        Raises:
            ValueError: If `rowIndices` and `colIndices` are both `None`.

        Returns:
            A `dict` where `key` is assay name and `value` is the subset matrices.
        """
        if rowIndices is None and colIndices is None:
            raise ValueError(
                "`rowIndices` and `colIndices` can not both be 'None'"
            )

        assay_subsets: Dict[str, Union[NDArray[Any], spmatrix]] = {}
        for name, assay in self._assays.items():
            if rowIndices is not None:
                assay = assay[rowIndices, :]  # type: ignore

            if colIndices is not None:
                assay = assay[:, colIndices]  # type: ignore

            assay_subsets[name] = assay  # type: ignore

        return assay_subsets

    @property
    def colnames(self) -> Sequence[str]:
        """Get column/sample names.

        Returns:
            A `list` of column names.
        """
        if isinstance(self._cols, DataFrame):
            return self._cols.index.tolist()  # type: ignore
        else:
            return self._cols.rowNames
