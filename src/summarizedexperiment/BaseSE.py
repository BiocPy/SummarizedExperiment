import warnings
from collections import OrderedDict
from typing import MutableMapping, Optional, Sequence, Tuple

import anndata
import pandas as pd
from biocframe import BiocFrame
from filebackedarray import H5BackedDenseData, H5BackedSparseData
from genomicranges import GenomicRanges

from ._type_checks import (
    is_bioc_or_pandas_frame,
    is_list_of_subclass,
    is_list_of_type,
    is_matrix_like,
)
from .dispatchers.colnames import get_colnames, set_colnames
from .dispatchers.rownames import get_rownames, set_rownames
from .types import (
    BiocOrPandasFrame,
    MatrixSlicerTypes,
    MatrixTypes,
    SlicerArgTypes,
    SlicerResult,
)
from .utils._combiners import (
    combine_assays,
    combine_frames,
    combine_metadata,
)
from .utils._slicer import get_indexes_from_bools, get_indexes_from_names

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche"
__license__ = "MIT"


class BaseSE:
    """Base class for Summarized Experiment.

    Represents genomic experiment data (`assays`), features (`rowdata`),
    sample data (`coldata`) and any other `metadata`.

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

    # TODO: should be an instance attribute
    _shape: Optional[Tuple] = None

    def __init__(
        self,
        assays: MutableMapping[str, MatrixTypes],
        rows: Optional[BiocOrPandasFrame] = None,
        cols: Optional[BiocOrPandasFrame] = None,
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
        assays: MutableMapping[str, MatrixTypes],
    ):
        """Internal method to validate experiment data (assays).

        Args:
            assays (MutableMapping[str, MatrixTypes]): experiment
                data.

        Raises:
            ValueError: when assays contain more than 2 dimensions.
            ValueError: if all assays do not have the same dimensions.
            TypeError: if assays includes a unsupported matrix representation.
        """

        for asy, mat in assays.items():
            if not is_matrix_like(mat):
                raise TypeError(
                    f"assay: {asy} is not a supported Matrix representation."
                )

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

    def _validate_rows(self, rows: BiocOrPandasFrame):
        """Internal method to validate feature information (rowdata).

        Args:
            rows (BiocOrPandasFrame): feature information (rowdata).

        Raises:
            ValueError: when number of rows does not match between rows & assays.
            TypeError: when rows is neither a pandas dataframe not Biocframe object.
        """
        if not is_bioc_or_pandas_frame(rows):
            raise TypeError(
                "rowData must be either a pandas `DataFrame` or a `BiocFrame`"
                f" object, provided {type(rows)}"
            )

        if rows.shape[0] != self._shape[0]:
            raise ValueError(
                f"Features and assays do not match. must be {self._shape[0]}"
                f" but provided {rows.shape[0]}"
            )

    def _validate_cols(self, cols: BiocOrPandasFrame):
        """Internal method to validate sample information (coldata).

        Args:
            cols (BiocOrPandasFrame): sample information (coldata).

        Raises:
            ValueError: when number of samples do not match between cols & assays.
            TypeError: when cols is neither a pandas dataframe not Biocframe object.
        """
        if not is_bioc_or_pandas_frame(cols):
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
    ) -> MutableMapping[str, MatrixTypes]:
        """Get assays.

        Returns:
            MutableMapping[str, MatrixTypes]: a dictionary with
            experiments names as keys and matrix data as values.
        """
        return self._assays

    @assays.setter
    def assays(
        self,
        assays: MutableMapping[str, MatrixTypes],
    ) -> None:
        """Set new experiment data (assays).

        Args:
            assays (MutableMapping[str, MatrixTypes]): new assays.
        """
        self._validate_assays(assays)
        self._assays = assays

    @property
    def rowData(self) -> BiocOrPandasFrame:
        """Get features.

        Returns:
            Optional[BiocOrPandasFrame]: features information.
        """
        return self._rows

    @rowData.setter
    def rowData(self, rows: BiocOrPandasFrame) -> None:
        """Set features.

        Args:
            rows (Optional[BiocOrPandasFrame]): new feature information.
        """
        rows = rows if rows is not None else BiocFrame({}, numberOfRows=self.shape[0])
        self._validate_rows(rows)
        self._rows = rows

    @property
    def colData(self) -> Optional[BiocOrPandasFrame]:
        """Get sample data.

        Returns:
            (BiocOrPandasFrame, optional): Sample information.
        """
        return self._cols

    @colData.setter
    def colData(self, cols: Optional[BiocOrPandasFrame]) -> None:
        """Set sample data.

        Args:
            cols (BiocOrPandasFrame, optional): sample data to update.
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
            f"  assays: {list(self.assays.keys())} \n"
            f"  features: {self.rowData.columns if self.rowData is not None else None} \n"
            f"  sample data: {self.colData.columns if self.colData is not None else None}"
        )
        return pattern

    def assay(self, name: str) -> MatrixTypes:
        """Convenience function to access an assay by name.

        Args:
            name (str): name of the assay.

        Raises:
            ValueError: if assay name does not exist.

        Returns:
            MatrixTypes: experiment data.
        """
        if name not in self._assays:
            raise ValueError(f"Assay {name} does not exist")

        return self._assays[name]

    def subsetAssays(
        self,
        rowIndices: Optional[MatrixSlicerTypes] = None,
        colIndices: Optional[MatrixSlicerTypes] = None,
    ) -> MutableMapping[str, MatrixTypes]:
        """Subset all assays to a slice (rows, cols).

        Args:
            rowIndices (MatrixSlicerTypes, optional): row indices to subset.
                Defaults to None.
            colIndices (MatrixSlicerTypes, optional): col indices to subset.
                Defaults to None.

        Raises:
            ValueError: if `rowIndices` and `colIndices` are both None.

        Returns:
            MutableMapping[str, MatrixTypes]: sliced experiment data.
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
        args: SlicerArgTypes,
    ) -> SlicerResult:
        """Internal method to slice `SE` by index.

        Args:
            args (SlicerArgTypes): indices or names to slice. tuple contains
                slices along dimensions (rows, cols).

        Raises:
            ValueError: Too many or too few slices provided.

        Returns:
            SlicerResult: sliced tuple.
        """

        if isinstance(args, tuple):
            if len(args) == 0:
                raise ValueError("Arguments must contain one slice")

            rowIndices = args[0]
            colIndices = None

            if len(args) > 1:
                colIndices = args[1]
            elif len(args) > 2:
                raise ValueError("contains too many slices")
        elif isinstance(args, list) or isinstance(args, slice):
            rowIndices = args
            colIndices = None
        else:
            raise ValueError(f"slicer {type(args)} is not supported")

        new_rows = self.rowData
        new_cols = self.colData
        new_assays = None

        if rowIndices is not None and self.rowData is not None:
            if is_list_of_type(rowIndices, str):
                rowIndices = get_indexes_from_names(
                    self.rowData.index, rowIndices
                )
            elif is_list_of_type(rowIndices, bool):
                if len(rowIndices) != self.shape[0]:
                    raise ValueError(
                        "since rowIndices is a boolean vector, its length should match "
                        f"the shape provided {len(rowIndices)}, must be {self.shape[0]}"
                    )
                rowIndices = get_indexes_from_bools(rowIndices)

            if is_list_of_type(rowIndices, int) or isinstance(rowIndices, slice):
                if isinstance(self.rowData, pd.DataFrame):
                    new_rows = new_rows.iloc[rowIndices]
                else:
                    new_rows = new_rows[rowIndices, :]
            else:
                raise TypeError("rowIndices not supported!")

        if colIndices is not None and self.colData is not None:
            if is_list_of_type(colIndices, str):
                colIndices = get_indexes_from_names(
                    self.colData.index, colIndices
                )
            elif is_list_of_type(colIndices, bool):
                if len(colIndices) != self.shape[1]:
                    raise ValueError(
                        "since colIndices is a boolean vector, its length should match "
                        f"the shape provided {len(colIndices)}, must be {self.shape[1]}"
                    )
                colIndices = get_indexes_from_bools(colIndices)

            if is_list_of_type(colIndices, int) or isinstance(colIndices, slice):
                if isinstance(self.colData, pd.DataFrame):
                    new_cols = new_cols.iloc[colIndices]
                else:
                    new_cols = new_cols[colIndices, :]
            else:
                raise TypeError("colIndices not supported!")

        new_assays = self.subsetAssays(rowIndices=rowIndices, colIndices=colIndices)

        return SlicerResult(new_rows, new_cols, new_assays, rowIndices, colIndices)

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
                    "while this is fine, this is currently not supported because "
                    "`AnnData` uses a transposed representation (cells X features) "
                    "rather than the bioconductor version (features X cells)"
                )

            layers[asy] = mat.transpose()

        trows = self.rowData
        if isinstance(self.rowData, GenomicRanges):
            trows = self.rowData.toPandas()

        obj = anndata.AnnData(
            obs=self.colData,
            var=trows,
            uns=self.metadata,
            layers=layers,
        )

        return obj

    def combineCols(
        self,
        *experiments: "BaseSE",
        useNames: bool = True,
        removeDuplicateColumns: bool = True,
    ) -> "BaseSE":
        """A more flexible version of `cbind`. Permits differences in the number
        and identity of rows, differences in `colData` fields, and even differences
        in the available `assays` among `SummarizedExperiment` objects being combined.

        Currently does not support range based merging of feature information when
        performing this operation.

        The row names of the resultant `SummarizedExperiment` object will
        simply be the row names of the first `SummarizedExperiment`.

        Note: if `removeDuplicateColumns` is True, we only keep the columns from this
        object (self). you can always do this operation later, but its useful when you
        are merging multiple summarized experiments and need to track metadata across
        objects.

        Args:
            experiments (BaseSE): `SummarizedExperiment`-like objects to concatenate.
            useNames (bool):
                - If `True`, then each input `SummarizedExperiment` must have non-null,
                non-duplicated row names. The row names of the resultant
                `SummarizedExperiment` object will be the union of the row names
                across all input objects.
                - If `False`, then each input `SummarizedExperiment` object must
                have the same number of rows.
            removeDuplicateColumns (bool): If `True`, remove any duplicate columns in
                `rowData` or `colData` of the resultant `SummarizedExperiment`. Defaults
                to `True`.

        Raises:
            TypeError:
                if any of the provided objects are not "SummarizedExperiment"-like.
            ValueError:
                - if there are null or duplicated row names (useNames=True)
                - if all objects do not have the same number of rows (useNames=False)

        Returns:
            BaseSE: combined `SummarizedExperiment` object.
        """

        if not is_list_of_subclass(experiments, BaseSE):
            raise TypeError(
                "not all provided objects are `SummarizedExperiment`-like objects"
            )

        ses = [self] + list(experiments)

        new_metadata = combine_metadata(experiments)

        all_coldata = [getattr(e, "colData") for e in ses]
        new_colData = combine_frames(
            all_coldata,
            axis=0,
            useNames=True,
            removeDuplicateColumns=removeDuplicateColumns,
        )

        all_rowdata = [getattr(e, "rowData") for e in ses]
        new_rowData = combine_frames(
            all_rowdata,
            axis=1,
            useNames=useNames,
            removeDuplicateColumns=removeDuplicateColumns,
        )

        new_assays = {}
        unique_assay_names = {assay_name for se in ses for assay_name in se.assayNames}
        for assay_name in unique_assay_names:
            merged_assays = combine_assays(
                assay_name=assay_name,
                experiments=ses,
                names=new_rowData.index,
                by="column",
                shape=(len(new_rowData), len(new_colData)),
                useNames=useNames,
            )
            new_assays[assay_name] = merged_assays

        current_class_const = type(self)
        return current_class_const(new_assays, new_rowData, new_colData, new_metadata)
