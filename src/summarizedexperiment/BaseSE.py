import warnings
from collections import OrderedDict
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple, Union

from biocframe import BiocFrame
from filebackedarray import H5BackedDenseData, H5BackedSparseData
from genomicranges import GenomicRanges
from pandas import DataFrame

from .dispatchers.colnames import get_colnames, set_colnames
from .dispatchers.rownames import get_rownames, set_rownames
from .type_checks import (
    is_bioc_or_pandas_frame,
    is_list_of_subclass,
    is_list_of_type,
    is_matrix_like,
)
from .types import (
    BiocOrPandasFrame,
    MatrixSlicerTypes,
    MatrixTypes,
    SlicerArgTypes,
    SlicerResult,
)
from .utils.combiners import (
    combine_assays,
    combine_frames,
    combine_metadata,
)
from .utils.slicer import get_indexes_from_bools, get_indexes_from_names

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche"
__license__ = "MIT"


class BaseSE:
    """Base class for `SummarizedExperiment`. Implements common properties and methods that can be reused across all
    derived classes.

    Container to represents genomic experiment data (`assays`), features (`row_data`),
    sample data (`col_data`) and any other `metadata`.

    Attributes:
        assays (MutableMapping[str, MatrixTypes]): Dictionary
            of matrices, with assay names as keys and 2-dimensional matrices represented as
            :py:class:`~numpy.ndarray` or :py:class:`scipy.sparse.spmatrix` matrices.

            Alternatively, you may use any 2-dimensional matrix that contains the property ``shape``
            and implements the slice operation using the ``__getitem__`` dunder method.

            All matrices in ``assays`` must be 2-dimensional and have the same
            shape (number of rows, number of columns).

        row_data (BiocOrPandasFrame, optional): Features, must be the same length as
            rows of the matrices in assays.

            Features may be either a :py:class:`~pandas.DataFrame` or
            :py:class:`~biocframe.BiocFrame.BiocFrame`.

            Defaults to None.
        col_data (BiocOrPandasFrame, optional): Sample data, must be
            the same length as columns of the matrices in assays.

            Sample Information may be either a :py:class:`~pandas.DataFrame` or
            :py:class:`~biocframe.BiocFrame.BiocFrame`.

            Defaults to None.
        metadata (MutableMapping, optional): Additional experimental metadata describing the
            methods. Defaults to None.
    """

    def __init__(
        self,
        assays: MutableMapping[str, MatrixTypes],
        rows: Optional[BiocOrPandasFrame] = None,
        cols: Optional[BiocOrPandasFrame] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize an instance of `BaseSE`."""

        self._shape: Optional[Tuple] = None

        if assays is None or not isinstance(assays, dict) or len(assays.keys()) == 0:
            raise Exception(
                "`assays` must be a dictionary and contain at least one matrix."
            )

        self._validate_assays(assays)
        self._assays = assays

        # should have _shape by now
        if self._shape is None:
            raise TypeError("This should not happen! `assays` is not consistent.")

        rows = (
            rows if rows is not None else BiocFrame({}, number_of_rows=self._shape[0])
        )
        self._validate_rows(rows)
        self._rows = rows

        cols = (
            cols if cols is not None else BiocFrame({}, number_of_rows=self._shape[1])
        )
        self._validate_cols(cols)
        self._cols = cols

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
            assays (MutableMapping[str, MatrixTypes]): Experiment
                data.

        Raises:
            ValueError: When ``assays`` contain more than 2 dimensions.
            ValueError: If all ``assays`` do not have the same dimensions.
            TypeError: If ``assays`` includes a unsupported matrix representation.
        """

        for asy, mat in assays.items():
            if not is_matrix_like(mat):
                raise TypeError(
                    f"Assay: '{asy}' is not a supported matrix representation."
                )

            if len(mat.shape) > 2:
                raise ValueError(
                    "Only 2-dimensional matrices are accepted, "
                    f"provided {len(mat.shape)} dimensions for `assay`: '{asy}'."
                )

            if self._shape is None:
                self._shape = mat.shape
                continue

            if mat.shape != self._shape:
                raise ValueError(
                    f"Assay: '{asy}' must be of shape '{self._shape}'"
                    f" but provided '{mat.shape}'."
                )

    def _validate_rows(self, rows: BiocOrPandasFrame):
        """Internal method to validate feature information (row_data).

        Args:
            rows (BiocOrPandasFrame): Feature data frame to validate.
                Features may be either a :py:class:`~pandas.DataFrame` or
                :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Raises:
            ValueError: When number of ``rows`` does not match between rows & assays.
            TypeError: When ``rows`` is neither a :py:class:`~pandas.DataFrame` nor
                :py:class:`~biocframe.BiocFrame.BiocFrame`.
        """
        if not is_bioc_or_pandas_frame(rows):
            raise TypeError(
                "`row_data` must be either a pandas `DataFrame` or a `BiocFrame`"
                f" object, provided {type(rows)}."
            )

        if rows.shape[0] != self._shape[0]:
            raise ValueError(
                f"`Features` and `assays` do not match. must be '{self._shape[0]}'"
                f" but provided '{rows.shape[0]}'."
            )

    def _validate_cols(self, cols: BiocOrPandasFrame):
        """Internal method to validate sample information (col_data).

        Args:
            cols (BiocOrPandasFrame): Sample information (col_data).
                Sample may be either a :py:class:`~pandas.DataFrame` or
                :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Raises:
            ValueError: When number of ``cols`` do not match between cols & assays.
            TypeError: When ``cols`` is neither a :py:class:`~pandas.DataFrame` nor
                :py:class:`~biocframe.BiocFrame.BiocFrame`.
        """
        if not is_bioc_or_pandas_frame(cols):
            raise TypeError(
                "`col_data` must be either a pandas `DataFrame` or a `BiocFrame`"
                f" object, provided {type(cols)}."
            )

        if cols.shape[0] != self._shape[1]:
            raise ValueError(
                f"`Sample` data and `assays` do not match. must be '{self._shape[1]}'"
                f" but provided '{cols.shape[0]}'."
            )

    @property
    def assays(
        self,
    ) -> Dict[str, MatrixTypes]:
        """Get all assays.

        Returns:
            Dict[str, MatrixTypes]: A dictionary with
            experiments names as keys and matrices as values.
        """
        return self._assays

    @assays.setter
    def assays(
        self,
        assays: MutableMapping[str, MatrixTypes],
    ):
        """Set new experiment data (assays).

        Args:
            assays (MutableMapping[str, MatrixTypes]): New assays.
        """
        self._validate_assays(assays)
        self._assays = assays

    @property
    def row_data(self) -> BiocOrPandasFrame:
        """Get features.

        Returns:
            BiocOrPandasFrame: Feature information.
        """
        return self._rows

    @row_data.setter
    def row_data(self, rows: Optional[BiocOrPandasFrame]):
        """Set features.

        Args:
            rows (BiocOrPandasFrame, optional): New feature information.
                If ``rows`` is None, an empty :py:class:`~biocframe.BiocFrame.BiocFrame`
                object is created.
        """
        rows = rows if rows is not None else BiocFrame({}, number_of_rows=self.shape[0])
        self._validate_rows(rows)
        self._rows = rows

    @property
    def col_data(self) -> BiocOrPandasFrame:
        """Get sample data.

        Returns:
            BiocOrPandasFrame: Sample information.
        """
        return self._cols

    @col_data.setter
    def col_data(self, cols: Optional[BiocOrPandasFrame]):
        """Set sample data.

        Args:
            cols (BiocOrPandasFrame, optional): New sample data.
                If ``cols`` is None, an empty :py:class:`~biocframe.BiocFrame.BiocFrame`
                object is created.
        """
        cols = cols if cols is not None else BiocFrame({}, number_of_rows=self.shape[1])

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
            Tuple[int, int]: A tuple with number of features and number of samples.
        """
        return self._shape

    @property
    def dims(self) -> Tuple[int, int]:
        """Dimensions of the experiment.

        Alias to :py:attr:`~summarizedexperiment.BaseSE.BaseSE.shape`.

        Returns:
            Tuple[int, int]: A tuple with number of features and samples.
        """
        return self.shape

    @property
    def assay_names(self) -> List[str]:
        """Get assay names.

        Returns:
            List[str]: List of assay names.
        """
        return list(self.assays.keys())

    @assay_names.setter
    def assay_names(self, names: Sequence[str]):
        """Replace all :py:attr:`~summarizedexperiment.BaseSE.BaseSE.assays`'s names.

        Args:
            names (Sequence[str]): New names.

        Raises:
            ValueError: If length of names does not match the number of assays.
        """
        current_names = self.assay_names
        if len(names) != len(current_names):
            raise ValueError("Provided `names` do not match number of `assays`.")

        new_assays = OrderedDict()
        for idx in range(len(names)):
            new_assays[names[idx]] = self._assays.pop(current_names[idx])

        self._assays = new_assays

    def __repr__(self) -> str:
        pattern = (
            f"Class BaseSE with {self.shape[0]} features and {self.shape[1]} samples \n"
            f"  assays: {', '.join(list(self.assays.keys()))} \n"
            f"  features: {self.row_data.columns if self.row_data is not None else None} \n"
            f"  sample data: {self.col_data.columns if self.col_data is not None else None}"
        )
        return pattern

    def assay(self, index_or_name: Union[int, str]) -> MatrixTypes:
        """Convenience function to access an :py:attr:`~summarizedexperiment.BaseSE.BaseSE.assays` by name.

        Alternatively, you may also provide an index position of the assay.

        Args:
            name (Union[int, str]): Name or index position of the assay.

        Raises:
            AttributeError: If assay name does not exist.
            IndexError: If index is greater than the number of assays.

        Returns:
            MatrixTypes: Experiment data.
        """
        if isinstance(index_or_name, int):
            if index_or_name < 0 or index_or_name > len(self.assay_names):
                raise IndexError("Index greater than the number of assays.")

            return self.assays[self.assay_names[index_or_name]]
        elif isinstance(index_or_name, str):
            if index_or_name not in self.assays:
                raise AttributeError(f"Assay: {index_or_name} does not exist.")

            return self.assays[index_or_name]

        raise TypeError(
            f"`index_or_name` must be a string or integer, provided {type(index_or_name)}."
        )

    def subset_assays(
        self,
        row_indices: Optional[MatrixSlicerTypes] = None,
        col_indices: Optional[MatrixSlicerTypes] = None,
    ) -> Dict[str, MatrixTypes]:
        """Subset all assays to a slice (rows, cols).

        If ``row_indices`` and ``col_indices`` are both None, a copy of the
        current assays is returned.

        Args:
            row_indices (MatrixSlicerTypes, optional): Row indices to subset.

                ``row_indices`` may be a list of integer indices to subset.

                Alternatively ``row_indices`` may be a boolean vector specifying
                True to keep the index or False to remove. Length of the boolean
                vector must match the number of rows in the experiment.

                Alternatively, ``row_indices`` may be a :py:class:`~slice` object.

                Defaults to None.

            col_indices (MatrixSlicerTypes, optional): Column indices to subset.

                ``col_indices`` may be a list of integer indices to subset.

                Alternatively ``col_indices`` may be a boolean vector specifying
                True to keep the index or False to remove. Length of the boolean
                vector must match the number of columns in the experiment.

                Alternatively, ``col_indices`` may be a :py:class:`~slice` object.

                Defaults to None.

        Raises:
            warning: If ``row_indices`` and ``col_indices`` are both None.

        Returns:
            Dict[str, MatrixTypes]: Sliced experiment data.
        """

        if row_indices is None and col_indices is None:
            warnings.warn("No slice is provided, this returns a copy of all assays!")
            return self.assays.copy()

        new_assays = OrderedDict()
        for asy, mat in self.assays.items():
            if row_indices is not None:
                mat = mat[row_indices, :]

            if col_indices is not None:
                mat = mat[:, col_indices]

            new_assays[asy] = mat

        return new_assays

    def _slice(
        self,
        args: SlicerArgTypes,
    ) -> SlicerResult:
        """Internal method to slice object by index.

        Args:
            args (SlicerArgTypes): Indices or names to slice. Tuple contains
                slices along dimensions (rows, cols).

                Each element in the tuple, might be either a integer vector (integer positions),
                boolean vector or :py:class:`~slice` object. Defaults to None.

        Raises:
            ValueError: Too many or too few slices provided.

        Returns:
            SlicerResult: Sliced tuple.
        """

        if isinstance(args, tuple):
            if len(args) == 0:
                raise ValueError("`args` must contain at least one slice.")

            row_indices = args[0]
            col_indices = None

            if len(args) > 1:
                col_indices = args[1]
            elif len(args) > 2:
                raise ValueError("`args` contains too many slices.")
        elif isinstance(args, list) or isinstance(args, slice):
            row_indices = args
            col_indices = None
        else:
            raise ValueError("`args` contains unsupported type.")

        new_rows = self.row_data
        new_cols = self.col_data
        new_assays = None

        if row_indices is not None and self.row_data is not None:
            if is_list_of_type(row_indices, str):
                row_indices = get_indexes_from_names(
                    get_rownames(self.row_data), row_indices
                )
            elif is_list_of_type(row_indices, bool):
                if len(row_indices) != self.shape[0]:
                    raise ValueError(
                        "`row_indices` is a boolean vector, length of vector must match the",
                        "number of rows.",
                    )
                row_indices = get_indexes_from_bools(row_indices)

            if is_list_of_type(row_indices, int) or isinstance(row_indices, slice):
                if isinstance(self.row_data, DataFrame):
                    new_rows = new_rows.iloc[row_indices]
                else:
                    new_rows = new_rows[row_indices, :]
            else:
                raise TypeError("`row_indices` is not supported!")

        if col_indices is not None and self.col_data is not None:
            if is_list_of_type(col_indices, str):
                col_indices = get_indexes_from_names(
                    get_rownames(self.col_data), col_indices
                )
            elif is_list_of_type(col_indices, bool):
                if len(col_indices) != self.shape[1]:
                    raise ValueError(
                        "`col_indices` is a boolean vector, length of vector must match the",
                        "number of columns.",
                    )
                col_indices = get_indexes_from_bools(col_indices)

            if is_list_of_type(col_indices, int) or isinstance(col_indices, slice):
                if isinstance(self.col_data, DataFrame):
                    new_cols = new_cols.iloc[col_indices]
                else:
                    new_cols = new_cols[col_indices, :]
            else:
                raise TypeError("`col_indices` not supported!")

        new_assays = self.subset_assays(
            row_indices=row_indices, col_indices=col_indices
        )

        return SlicerResult(new_rows, new_cols, new_assays, row_indices, col_indices)

    @property
    def row_names(self) -> List[str]:
        """Get row/feature index.

        Returns:
            List[str]: List of row names.
        """
        return get_rownames(self.row_data)

    @row_names.setter
    def row_names(self, names: Sequence[str]):
        """Set row/feature names for the experiment.

        Args:
            names (Sequence[str]): New feature names.

        Raises:
            ValueError: Length of ``names`` must be the same as number of rows.
        """
        if len(names) != self.shape[0]:
            raise ValueError("Length of `names` must be the same as number of rows.")

        self._rows = set_rownames(self.row_data, names)

    @property
    def colnames(self) -> List[str]:
        """Get column/sample names.

        Returns:
            List[str]: List of sample names.
        """
        return get_colnames(self.col_data)

    @colnames.setter
    def colnames(self, names: Sequence[str]):
        """Set column/sample names for the experiment.

        Args:
            names (Sequence[str]): New samples names.

        Raises:
            ValueError: Length of ``names`` must be the same as number of columns.
        """
        if len(names) != self.shape[1]:
            raise ValueError("Length of `names` must be the same as number of columns.")

        self._cols = set_colnames(self.col_data, names)

    def to_anndata(
        self,
    ) -> "AnnData":
        """Transform :py:class:`summarizedexperiment.BaseSE`-like to :py:class:`~anndata.AnnData` representation.

        Returns:
            AnnData: An `AnnData` representation of the experiment..
        """
        from anndata import AnnData

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            if isinstance(mat, H5BackedDenseData) or isinstance(
                mat, H5BackedSparseData
            ):
                raise ValueError(
                    f"Assay: '{asy}' is not supported. Uses a file backed representation."
                )

            layers[asy] = mat.transpose()

        trows = self.row_data
        if isinstance(self.row_data, GenomicRanges):
            trows = self.row_data.to_pandas()

        obj = AnnData(
            obs=self.col_data,
            var=trows,
            uns=self.metadata,
            layers=layers,
        )

        return obj

    def combine_cols(
        self,
        *experiments: "BaseSE",
        use_names: bool = True,
        remove_duplicate_columns: bool = True,
    ) -> "BaseSE":
        """A more flexible version of ``cbind``. Permits differences in the number and identity of rows, differences in
        :py:attr:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.col_data` fields, and even differences
        in the available `assays` among :py:class:`~summarizedexperiment.SummarizedExperiment.BaseSE`-derived objects
        being combined.

        Currently does not support range based merging of feature information when
        performing this operation.

        The row names of the resultant `SummarizedExperiment` object will
        simply be the row names of the first `SummarizedExperiment`.

        Note: if `remove_duplicate_columns` is True, we only keep the columns from this
        object (self). you can always do this operation later, but its useful when you
        are merging multiple summarized experiments and need to track metadata across
        objects.

        Args:
            experiments (BaseSE): `SummarizedExperiment`-like objects to concatenate.

            use_names (bool):

                - If `True`, then each input `SummarizedExperiment` must have non-null,
                non-duplicated row names. The row names of the resultant
                `SummarizedExperiment` object will be the union of the row names
                across all input objects.
                - If `False`, then each input `SummarizedExperiment` object must
                have the same number of rows.

            remove_duplicate_columns (bool): If `True`, remove any duplicate columns in
                `row_data` or `col_data` of the resultant `SummarizedExperiment`. Defaults
                to `True`.

        Raises:
            TypeError:
                If any of the provided objects are not "SummarizedExperiment"-like.
            ValueError:
                - If there are null or duplicated row names (use_names=True)
                - If all objects do not have the same number of rows (use_names=False)

        Returns:
            Same type as the caller with the combined experiments.
        """

        if not is_list_of_subclass(experiments, BaseSE):
            raise TypeError(
                "Not all provided objects are `SummarizedExperiment`-like objects."
            )

        ses = [self] + list(experiments)

        new_metadata = combine_metadata(experiments)

        all_col_data = [getattr(e, "col_data") for e in ses]
        new_col_data = combine_frames(
            all_col_data,
            axis=0,
            use_names=True,
            remove_duplicate_columns=remove_duplicate_columns,
        )

        all_row_data = [getattr(e, "row_data") for e in ses]
        new_row_data = combine_frames(
            all_row_data,
            axis=1,
            use_names=use_names,
            remove_duplicate_columns=remove_duplicate_columns,
        )

        new_assays = {}
        unique_assay_names = {assay_name for se in ses for assay_name in se.assay_names}
        for assay_name in unique_assay_names:
            merged_assays = combine_assays(
                assay_name=assay_name,
                experiments=ses,
                names=new_row_data.index,
                by="column",
                shape=(len(new_row_data), len(new_col_data)),
                use_names=use_names,
            )
            new_assays[assay_name] = merged_assays

        current_class_const = type(self)
        return current_class_const(new_assays, new_row_data, new_col_data, new_metadata)
