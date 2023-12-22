import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import biocutils as ut
from biocframe import BiocFrame
from genomicranges import GenomicRanges

from ._frameutils import _sanitize_frame
from .type_checks import is_matrix_like
from .types import SliceResult

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche"
__license__ = "MIT"


def _guess_assay_shape(assays, rows, cols) -> tuple:
    _keys = list(assays.keys())
    if len(_keys) > 0:
        _first = _keys[0]
        return assays[_first].shape

    _r = 0
    if rows is not None:
        _r = rows.shape[0]

    _c = 0
    if cols is not None:
        _c = cols.shape[0]

    return (_r, _c)


def _validate_assays(assays, shape) -> tuple:
    if assays is None or not isinstance(assays, dict) or len(assays.keys()) == 0:
        raise Exception(
            "`assays` must be a dictionary and contain atleast one 2-dimensional matrix."
        )

    for asy, mat in assays.items():
        if not is_matrix_like(mat):
            raise TypeError(f"Assay: '{asy}' is not a supported matrix representation.")

        if len(mat.shape) > 2:
            raise ValueError(
                "Only 2-dimensional matrices are accepted, "
                f"provided {len(mat.shape)} dimensions for `assay`: '{asy}'."
            )

        if shape is None:
            shape = mat.shape
            continue

        if mat.shape != shape:
            raise ValueError(
                f"Assay: '{asy}' must be of shape '{shape}'"
                f" but provided '{mat.shape}'."
            )


def _validate_rows(rows, shape):
    if not isinstance(rows, BiocFrame):
        raise TypeError("'row_data' is not a `BiocFrame` object.")

    if rows.shape[0] != shape[0]:
        raise ValueError(
            f"Number of features mismatch with number of rows in assays. Must be '{shape[0]}'"
            f" but provided '{rows.shape[0]}'."
        )


def _validate_cols(cols, shape):
    if not isinstance(cols, BiocFrame):
        raise TypeError("'col_data' is not a `BiocFrame` object.")

    if cols.shape[0] != shape[1]:
        raise ValueError(
            f"Number of samples mismatch with number of columns in assays. Must be '{shape[1]}'"
            f" but provided '{cols.shape[0]}'."
        )


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise TypeError("'metadata' should be a dictionary")


class BaseSE:
    """Base class for ``SummarizedExperiment``. This class provides common properties and methods that can be utilized
    across all derived classes.

    This container represents genomic experiment data in the form of
    ``assays``, features in ``row_data``, sample data in ``col_data``,
    and any other relevant ``metadata``.
    """

    def __init__(
        self,
        assays: Dict[str, Any],
        row_data: Optional[BiocFrame] = None,
        col_data: Optional[BiocFrame] = None,
        metadata: Optional[dict] = None,
        validate: bool = True,
    ) -> None:
        """Initialize an instance of ``BaseSE``.

        Args:
            assays:
                A dictionary containing matrices, with assay names as keys
                and 2-dimensional matrices represented as either
                :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

                Alternatively, you may use any 2-dimensional matrix that has
                the ``shape`` property and implements the slice operation
                using the ``__getitem__`` dunder method.

                All matrices in assays must be 2-dimensional and have the
                same shape (number of rows, number of columns).

            rows:
                Features, must be the same length as the number of rows of
                the matrices in assays.

                Feature information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            cols:
                Sample data, must be the same length as the number of
                columns of the matrices in assays.

                Sample information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            metadata:
                Additional experimental metadata describing the methods.
                Defaults to None.

            validate:
                Internal use only.
        """
        self._assays = assays

        self._shape = _guess_assay_shape(assays, row_data, col_data)
        self._rows = _sanitize_frame(row_data, self._shape[0])
        self._cols = _sanitize_frame(col_data, self._shape[1])
        self._metadata = metadata if metadata is not None else {}

        if validate:
            _validate_assays(self._assays, self._shape)

            if self._shape is None:
                raise RuntimeError("Cannot extract 'shape' from assays!")

            _validate_rows(self._rows, self._shape)
            _validate_cols(self._cols, self._shape)
            _validate_metadata(self._metadata)

    def _define_output(self, in_place: bool = False) -> "BaseSE":
        if in_place is True:
            return self
        else:
            return self.__copy__()

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``BaseSE``.
        """
        from copy import deepcopy

        _assays_copy = deepcopy(self._assays)
        _rows_copy = deepcopy(self._rows)
        _cols_copy = deepcopy(self._cols)
        _metadata_copy = deepcopy(self.metadata)

        current_class_const = type(self)
        return current_class_const(
            assays=_assays_copy,
            row_data=_rows_copy,
            col_data=_cols_copy,
            metadata=_metadata_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``BaseSE``.
        """
        current_class_const = type(self)
        return current_class_const(
            assays=self._assays,
            row_data=self._rows,
            col_data=self._cols,
            metadata=self._metadata,
        )

    def copy(self):
        """Alias for :py:meth:`~__copy__`."""
        return self.__copy__()

    ######################################
    ######>> length and iterators <<######
    ######################################

    def __len__(self) -> int:
        """
        Returns:
            Number of rows.
        """
        return self.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the experiment.

        Returns:
            Tuple[int, int]: A tuple (m,n),
            where `m` is the number of features/rows, and
            `n` is the number of samples/columns.
        """
        return self._shape

    @property
    def dims(self) -> Tuple[int, int]:
        """Alias to :py:attr:`~summarizedexperiment.BaseSE.BaseSE.shape`.

        Returns:
            Tuple[int, int]: A tuple (m,n),
            where `m` is the number of features/rows, and
            `n` is the number of samples/columns.
        """
        return self.shape

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self) -> str:
        pattern = (
            f"Class {type(self).__name__} with {self.shape[0]} features and {self.shape[1]} samples \n"
            f"  assays: {', '.join(list(self.assays.keys()))} \n"
            f"  row_data: {self._rows.names if self._rows is not None else None} \n"
            f"  col_data: {self._cols.names if self._cols is not None else None}"
        )
        return pattern

    ########################
    ######>> assays <<######
    ########################

    def get_assays(self) -> Dict[str, Any]:
        """Access assays/experimental data.

        Returns:
            A dictionary with keys as assay names and value
            the experimental data.
        """
        return self._assays

    def set_assays(self, assays: Dict[str, Any], in_place: bool = False):
        """Set new experiment data (assays).

        Args:
            assays:
                New assays.

            in_place:
                Whether to modify the ``BaseSE`` in place.

        Returns:
            A modified ``BaseSE`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """

        _validate_assays(assays, self._shape)
        output = self._define_output(in_place)
        output._assays = assays
        return output

    @property
    def assays(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_assays`."""
        return self.get_assays()

    @assays.setter
    def assays(self, assays: Dict[str, Any]):
        """Alias for :py:meth:`~set_assays` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'assays' is an in-place operation, use 'set_assays' instead",
            UserWarning,
        )
        self.set_assays(assays, in_place=True)

    ##########################
    ######>> row_data <<######
    ##########################

    def get_rowdata(self) -> BiocFrame:
        """Get features.

        Returns:
            Feature information.
        """
        return self._rows

    def set_rowdata(self, rows: Optional[BiocFrame], in_place: bool = False):
        """Set new feature information.

        Args:
            rows:
                New feature information.

                If ``rows`` is None, an empty
                :py:class:`~biocframe.BiocFrame.BiocFrame`
                object is created.

            in_place:
                Whether to modify the ``BaseSE`` in place.

        Returns:
            A modified ``BaseSE`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        rows = _sanitize_frame(rows, self._shape[0])
        _validate_rows(rows, self._shape)

        output = self._define_output(in_place)
        output._rows = rows
        return output

    @property
    def rowdata(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_rowdata`."""
        return self.get_rowdata()

    @rowdata.setter
    def rowdata(self, rows: Optional[BiocFrame]):
        """Alias for :py:meth:`~set_rowdata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'rowdata' is an in-place operation, use 'set_rowdata' instead",
            UserWarning,
        )
        self.set_rowdata(rows, in_place=True)

    @property
    def row_data(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_rowdata`."""
        return self.get_rowdata()

    @row_data.setter
    def row_data(self, rows: Optional[BiocFrame]):
        """Alias for :py:meth:`~set_rowdata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'rowdata' is an in-place operation, use 'set_rowdata' instead",
            UserWarning,
        )
        self.set_rowdata(rows, in_place=True)

    ##########################
    ######>> col_data <<######
    ##########################

    def get_coldata(self) -> BiocFrame:
        """Get sample data.

        Returns:
            Sample information.
        """
        return self._cols

    def set_coldata(self, cols: Optional[BiocFrame], in_place: bool = False):
        """Set sample data.

        Args:
            cols:
                New sample data.

                If ``cols`` is None, an empty
                :py:class:`~biocframe.BiocFrame.BiocFrame`
                object is created.

            in_place:
                Whether to modify the ``BaseSE`` in place.

        Returns:
            A modified ``BaseSE`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        cols = _sanitize_frame(cols, self._shape[1])
        _validate_cols(cols, self._shape)

        output = self._define_output(in_place)
        output._cols = cols
        return output

    @property
    def coldata(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_coldata`."""
        return self.get_coldata()

    @coldata.setter
    def coldata(self, rows: Optional[BiocFrame]):
        """Alias for :py:meth:`~set_coldata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'coldata' is an in-place operation, use 'set_coldata' instead",
            UserWarning,
        )
        self.set_coldata(rows, in_place=True)

    @property
    def col_data(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_coldata`."""
        return self.get_coldata()

    @col_data.setter
    def col_data(self, rows: Optional[BiocFrame]):
        """Alias for :py:meth:`~set_coldata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'coldata' is an in-place operation, use 'set_coldata' instead",
            UserWarning,
        )
        self.set_coldata(rows, in_place=True)

    ###########################
    ######>> metadata <<#######
    ###########################

    def get_metadata(self) -> dict:
        """
        Returns:
            Dictionary of metadata for this object.
        """
        return self._metadata

    def set_metadata(self, metadata: dict, in_place: bool = False) -> "GenomicRanges":
        """Set additional metadata.

        Args:
            metadata:
                New metadata for this object.

            in_place:
                Whether to modify the ``BaseSE`` in place.

        Returns:
            A modified ``BaseSE`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        if not isinstance(metadata, dict):
            raise TypeError(
                f"`metadata` must be a dictionary, provided {type(metadata)}."
            )
        output = self._define_output(in_place)
        output._metadata = metadata
        return output

    @property
    def metadata(self) -> dict:
        """Alias for :py:attr:`~get_metadata`."""
        return self.get_metadata()

    @metadata.setter
    def metadata(self, metadata: dict):
        """Alias for :py:attr:`~set_metadata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'metadata' is an in-place operation, use 'set_metadata' instead",
            UserWarning,
        )
        self.set_metadata(metadata, in_place=True)

    #############################
    ######>> assay names <<######
    #############################

    def get_assay_names(self) -> List[str]:
        """Get assay names.

        Returns:
            List of assay names.
        """
        return list(self.assays.keys())

    def set_assay_names(self, names: List[str], in_place: bool = False):
        """Replace :py:attr:`~summarizedexperiment.BaseSE.BaseSE.assays`'s names.

        Args:
            names:
                New names.

            in_place:
                Whether to modify the ``BaseSE`` in place.

        Returns:
            A modified ``BaseSE`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        current_names = self.assay_names
        if len(names) != len(current_names):
            raise ValueError("Length of 'names' does not match the number of `assays`.")

        new_assays = OrderedDict()
        for idx in range(len(names)):
            new_assays[names[idx]] = self._assays.pop(current_names[idx])

        output = self._define_output(in_place)
        output._assays = new_assays
        return output

    @property
    def assay_names(self) -> List[str]:
        """Alias for :py:attr:`~get_assay_names`."""
        return self.get_assay_names()

    @assay_names.setter
    def assay_names(self, names: List[str]):
        """Alias for :py:attr:`~set_assay_names` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'assay_names' is an in-place operation, use 'set_assay_names' instead",
            UserWarning,
        )
        self.set_assay_names(names, in_place=True)

    ################################
    ######>> assay getters <<#######
    ################################

    def assay(self, assay: Union[int, str]) -> Any:
        """Convenience method to access an :py:attr:`~summarizedexperiment.BaseSE.BaseSE.assays` by name or index.

        Args:
            assay:
                Name or index position of the assay.

        Raises:
            AttributeError:
                If the assay name does not exist.
            IndexError:
                If index is greater than the number of assays.

        Returns:
            Experiment data.
        """
        if isinstance(assay, int):
            if assay < 0 or assay > len(self.assay_names):
                raise IndexError("Index greater than the number of assays.")

            return self.assays[self.assay_names[assay]]
        elif isinstance(assay, str):
            if assay not in self.assays:
                raise AttributeError(f"Assay: {assay} does not exist.")

            return self.assays[assay]

        raise TypeError(f"'assay' must be a string or integer, provided {type(assay)}.")

    ##########################
    ######>> slicers <<#######
    ##########################

    def _normalize_row_slice(self, rows: Union[str, int, bool, Sequence]):
        _scalar = None
        if rows != slice(None):
            rows, _scalar = ut.normalize_subscript(
                rows, len(self._rows), self._rows.row_names
            )

        return rows, _scalar

    def _normalize_column_slice(self, columns: Union[str, int, bool, Sequence]):
        _scalar = None
        if columns != slice(None):
            columns, _scalar = ut.normalize_subscript(
                columns, len(self._cols), self._cols.row_names
            )

        return columns, _scalar

    def subset_assays(
        self,
        rows: Optional[Union[str, int, bool, Sequence]],
        columns: Optional[Union[str, int, bool, Sequence]],
    ) -> Dict[str, Any]:
        """Subset all assays by the slice defined by rows and columns.

        If both ``row_indices`` and ``col_indices`` are None, a shallow copy of the
        current assays is returned.

        Args:
            rows:
                Row indices to subset.

                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted, see
                :py:meth:`~biocutils.normalize_subscript.normalize_subscript`.

            columns:
                Column indices to subset.

                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted, see
                :py:meth:`~biocutils.normalize_subscript.normalize_subscript`.

        Returns:
            Sliced experiment data.
        """

        if rows is None and columns is None:
            warnings.warn("No slice is provided, this returns a copy of all assays!")
            return self.assays.copy()

        if rows is None:
            rows = slice(None)

        if columns is None:
            columns = slice(None)

        rows, _ = self._normalize_row_slice(rows)
        columns, _ = self._normalize_column_slice(columns)

        new_assays = OrderedDict()
        for asy, mat in self.assays.items():
            if rows != slice(None):
                mat = mat[rows, :]

            if columns != slice(None):
                mat = mat[:, columns]

            new_assays[asy] = mat

        return new_assays

    def _generic_slice(
        self,
        rows: Optional[Union[str, int, bool, Sequence]],
        columns: Optional[Union[str, int, bool, Sequence]],
    ) -> SliceResult:
        """Slice ``SummarizedExperiment`` along the rows and/or columns, based on their indices or names.

        Args:
            rows:
                Rows to be extracted.

                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted, see
                :py:meth:`~biocutils.normalize_subscript.normalize_subscript`.

            columns:
                Columns to be extracted.

                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted, see
                :py:meth:`~biocutils.normalize_subscript.normalize_subscript`.

        Returns:
            The sliced tuple containing the new rows, columns, assays and realized indices
            for use in downstream methods.
        """

        new_rows = self.row_data
        new_cols = self.col_data
        new_assays = {}

        if rows is None:
            rows = slice(None)

        if columns is None:
            columns = slice(None)

        if rows is not None:
            rows, _ = self._normalize_row_slice(rows=rows)
            new_rows = new_rows[rows, :]

        if columns is not None and self.col_data is not None:
            columns, _ = self._normalize_column_slice(columns=columns)
            new_cols = new_cols[columns, :]

        new_assays = self.subset_assays(rows=rows, columns=columns)

        return SliceResult(new_rows, new_cols, new_assays, rows, columns)

    def get_slice(
        self,
        rows: Optional[Union[str, int, bool, Sequence]],
        columns: Optional[Union[str, int, bool, Sequence]],
    ) -> "BaseSE":
        """Alias for :py:attr:`~__getitem__`, for back-compatibility."""

        slicer = self._generic_slice(rows=rows, columns=columns)

        current_class_const = type(self)
        return current_class_const(
            assays=slicer.assays,
            row_data=slicer.rows,
            col_data=slicer.columns,
            metadata=self._metadata,
        )

    def __getitem__(
        self,
        args: Union[int, str, Sequence, tuple],
    ) -> "BaseSE":
        """Subset a `SummarizedExperiment`.

        Args:
            args:
                Integer indices, a boolean filter, or (if the current object is
                named) names specifying the ranges to be extracted, see
                :py:meth:`~biocutils.normalize_subscript.normalize_subscript`.

                Alternatively a tuple of length 1. The first entry specifies
                the rows to retain based on their names or indices.

                Alternatively a tuple of length 2. The first entry specifies
                the rows to retain, while the second entry specifies the
                columns to retain, based on their names or indices.

        Raises:
            ValueError: If too many or too few slices provided.

        Returns:
            Same type as caller with the sliced rows and columns.
        """
        if isinstance(args, (str, int)):
            return self.get_slice(args, slice(None))

        if isinstance(args, tuple):
            if len(args) == 0:
                raise ValueError("At least one slicing argument must be provided.")

            if len(args) == 1:
                return self.get_slice(args[0], slice(None))
            elif len(args) == 2:
                return self.get_slice(args[0], args[1])
            else:
                raise ValueError(
                    f"`{type(self).__name__}` only supports 2-dimensional slicing."
                )

        raise TypeError(
            "args must be a sequence or a scalar integer or string or a tuple of atmost 2 values."
        )

    ###############################
    ######>> names accessor <<#####
    ###############################

    def get_row_names(self) -> Optional[ut.Names]:
        """
        Returns:
            List of row names, or None if no row names are available.
        """
        return self._rows.get_row_names()

    @property
    def rownames(self) -> List[str]:
        """Alias for :py:attr:`~get_rownames`, provided for back-compatibility."""
        return self.get_row_names()

    def get_column_names(self) -> Optional[ut.Names]:
        """
        Returns:
            List of column names, or None if no column names are available.
        """
        return self._cols.get_row_names()

    @property
    def colnames(self) -> List[str]:
        """Alias for :py:attr:`~get_rownames`, provided for back-compatibility."""
        return self.get_column_names()

    ################################
    ######>> AnnData interop <<#####
    ################################

    def to_anndata(self):
        """Transform :py:class:`summarizedexperiment.BaseSE`-like into a :py:class:`~anndata.AnnData` representation.

        Returns:
            An `AnnData` representation of the experiment.
        """
        from anndata import AnnData

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()

        trows = self._rows.to_pandas()

        obj = AnnData(
            obs=self._cols.to_pandas(),
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
