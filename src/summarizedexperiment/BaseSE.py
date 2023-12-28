import warnings
from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import biocframe
import biocutils as ut

from ._frameutils import _sanitize_frame
from .type_checks import is_matrix_like

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche"
__license__ = "MIT"

SliceResult = namedtuple(
    "SlicerResult",
    [
        "rows",
        "columns",
        "assays",
        "row_names",
        "column_names",
        "row_indices",
        "col_indices",
    ],
)


def _guess_assay_shape(assays, rows, cols, row_names, col_names) -> tuple:
    _keys = list(assays.keys())
    if len(_keys) > 0:
        _first = _keys[0]
        return assays[_first].shape

    _r = 0
    if rows is not None:
        _r = rows.shape[0]
    elif row_names is not None:
        _r = len(row_names)

    _c = 0
    if cols is not None:
        _c = cols.shape[0]
    elif col_names is not None:
        _c = len(col_names)

    return (_r, _c)


def _validate_assays(assays, shape) -> tuple:
    if assays is None or not isinstance(assays, dict):  # or len(assays.keys()) == 0
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


def _validate_rows(rows, names, shape):
    if not isinstance(rows, biocframe.BiocFrame):
        raise TypeError("'row_data' is not a `BiocFrame` object.")

    if rows.shape[0] != shape[0]:
        raise ValueError(
            f"Number of features ('row_data') mismatch with number of rows in assays. Must be '{shape[0]}'"
            f" but provided '{rows.shape[0]}'."
        )

    if names is not None:
        if len(names) != shape[0]:
            raise ValueError(
                f"Length of 'row_names' mismatch with number of rows. Must be '{shape[0]}'"
                f" but provided '{len(names)}'."
            )


def _validate_cols(cols, names, shape):
    if not isinstance(cols, biocframe.BiocFrame):
        raise TypeError("'column_data' is not a `BiocFrame` object.")

    if cols.shape[0] != shape[1]:
        raise ValueError(
            f"Number of samples ('column_data') mismatch with number of columns in assays. Must be '{shape[1]}'"
            f" but provided '{cols.shape[0]}'."
        )

    if names is not None:
        if len(names) != shape[1]:
            raise ValueError(
                f"Length of 'column_names' mismatch with number of columns. Must be '{shape[1]}'"
                f" but provided '{len(names)}'."
            )


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise TypeError("'metadata' should be a dictionary")


class BaseSE:
    """Base class for ``SummarizedExperiment``. This class provides common properties and methods that can be utilized
    across all derived classes.

    This container represents genomic experiment data in the form of
    ``assays``, features in ``row_data``, sample data in ``column_data``,
    and any other relevant ``metadata``.
    """

    def __init__(
        self,
        assays: Dict[str, Any] = None,
        row_data: Optional[biocframe.BiocFrame] = None,
        column_data: Optional[biocframe.BiocFrame] = None,
        row_names: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
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

            row_names:
                A list of strings, same as the number of rows.Defaults to None.

            column_names:
                A list of string, same as the number of columns. Defaults to None.

            metadata:
                Additional experimental metadata describing the methods.
                Defaults to None.

            validate:
                Internal use only.
        """
        self._assays = assays if assays is not None else {}

        self._shape = _guess_assay_shape(
            self._assays, row_data, column_data, row_names, column_names
        )

        if self._shape is None:
            raise RuntimeError(
                "Failed to guess the 'shape' from the provided parameters!"
            )

        self._rows = _sanitize_frame(row_data, self._shape[0])
        self._cols = _sanitize_frame(column_data, self._shape[1])

        if row_names is not None and not isinstance(row_names, ut.Names):
            row_names = ut.Names(row_names)
        self._row_names = row_names

        if column_names is not None and not isinstance(column_names, ut.Names):
            column_names = ut.Names(column_names)
        self._column_names = column_names

        self._metadata = metadata if metadata is not None else {}

        if validate:
            _validate_assays(self._assays, self._shape)

            if self._shape is None:
                raise RuntimeError("Cannot guess 'shape' from assays!")

            _validate_rows(self._rows, self._row_names, self._shape)
            _validate_cols(self._cols, self._column_names, self._shape)
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
        _row_names_copy = deepcopy(self._row_names)
        _col_names_copy = deepcopy(self._column_names)

        current_class_const = type(self)
        return current_class_const(
            assays=_assays_copy,
            row_data=_rows_copy,
            column_data=_cols_copy,
            row_names=_row_names_copy,
            column_names=_col_names_copy,
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
            column_data=self._cols,
            row_names=self._row_names,
            column_names=self._column_names,
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
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}(number_of_rows={self.shape[0]}"
        output += f", number_of_columns={self.shape[1]}"
        output += ", assays=" + ut.print_truncated_list(self.assay_names)
        output += ", row_data=" + self._rows.__repr__()
        output += ", column_data=" + self._cols.__repr__()

        if self._row_names is not None:
            output += ", row_names=" + ut.print_truncated_list(self._row_names)

        if self._column_names is not None:
            output += ", column_names=" + ut.print_truncated_list(self._column_names)

        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)

        output += ")"
        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"

        output += f"dimensions: ({self.shape[0]}, {self.shape[1]})\n"

        output += f"assays({len(self.assay_names)}): {ut.print_truncated_list(self.assay_names)}\n"

        output += f"row_data columns({len(self._rows.column_names)}): {ut.print_truncated_list(self._rows.column_names)}\n"
        output += f"row_names({0 if self._row_names is None else len(self._row_names)}): {' ' if self._row_names is None else ut.print_truncated_list(self._row_names)}\n"

        output += f"column_data columns({len(self._cols.column_names)}): {ut.print_truncated_list(self._cols.column_names)}\n"
        output += f"column_names({0 if self._column_names is None else len(self._column_names)}): {' ' if self._column_names is None else ut.print_truncated_list(self._column_names)}\n"

        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

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

    def set_assays(self, assays: Dict[str, Any], in_place: bool = False) -> "BaseSE":
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

    def get_row_data(self) -> biocframe.BiocFrame:
        """Get features.

        Returns:
            Feature information.
        """
        return self._rows

    def set_row_data(
        self, rows: Optional[biocframe.BiocFrame], in_place: bool = False
    ) -> "BaseSE":
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
        _validate_rows(rows, self._row_names, self._shape)

        output = self._define_output(in_place)
        output._rows = rows
        return output

    @property
    def rowdata(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_rowdata`."""
        return self.get_row_data()

    @rowdata.setter
    def rowdata(self, rows: Optional[biocframe.BiocFrame]):
        """Alias for :py:meth:`~set_rowdata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'rowdata' is an in-place operation, use 'set_rowdata' instead",
            UserWarning,
        )
        self.set_row_data(rows, in_place=True)

    @property
    def row_data(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_rowdata`."""
        return self.get_row_data()

    @row_data.setter
    def row_data(self, rows: Optional[biocframe.BiocFrame]):
        """Alias for :py:meth:`~set_rowdata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'rowdata' is an in-place operation, use 'set_rowdata' instead",
            UserWarning,
        )
        self.set_row_data(rows, in_place=True)

    ##########################
    ######>> col_data <<######
    ##########################

    def get_column_data(self) -> biocframe.BiocFrame:
        """Get sample data.

        Returns:
            Sample information.
        """
        return self._cols

    def set_column_data(
        self, cols: Optional[biocframe.BiocFrame], in_place: bool = False
    ) -> "BaseSE":
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
        _validate_cols(cols, self._column_names, self._shape)

        output = self._define_output(in_place)
        output._cols = cols
        return output

    @property
    def columndata(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_coldata`."""
        return self.get_column_data()

    @columndata.setter
    def columndata(self, cols: Optional[biocframe.BiocFrame]):
        """Alias for :py:meth:`~set_coldata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'coldata' is an in-place operation, use 'set_columndata' instead",
            UserWarning,
        )
        self.set_column_data(cols, in_place=True)

    @property
    def coldata(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_coldata`."""
        return self.get_column_data()

    @coldata.setter
    def coldata(self, cols: Optional[biocframe.BiocFrame]):
        """Alias for :py:meth:`~set_coldata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'coldata' is an in-place operation, use 'set_columndata' instead",
            UserWarning,
        )
        self.set_column_data(cols, in_place=True)

    @property
    def column_data(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_coldata`."""
        return self.get_column_data()

    @column_data.setter
    def column_data(self, cols: Optional[biocframe.BiocFrame]):
        """Alias for :py:meth:`~set_coldata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'coldata' is an in-place operation, use 'set_coldata' instead",
            UserWarning,
        )
        self.set_column_data(cols, in_place=True)

    @property
    def col_data(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_coldata`."""
        return self.get_column_data()

    @col_data.setter
    def col_data(self, cols: Optional[biocframe.BiocFrame]):
        """Alias for :py:meth:`~set_coldata` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'coldata' is an in-place operation, use 'set_columndata' instead",
            UserWarning,
        )
        self.set_column_data(cols, in_place=True)

    ##########################
    ######>> row names <<#####
    ##########################

    def get_row_names(self) -> Optional[ut.Names]:
        """
        Returns:
            List of row names, or None if no row names are available.
        """
        return self._row_names

    def set_row_names(
        self, names: Optional[List[str]], in_place: bool = False
    ) -> "BaseSE":
        """Set new row names.

        Args:
            names:
                New names, same as the number of rows.

                May be `None` to remove row names.

            in_place:
                Whether to modify the ``BaseSE`` in place.

        Returns:
            A modified ``BaseSE`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        if names is not None and not isinstance(names, ut.Names):
            names = ut.Names(names)

        _validate_rows(self._rows, names, self.shape)

        output = self._define_output(in_place)
        output._row_names = names
        return output

    @property
    def rownames(self) -> Optional[ut.Names]:
        """Alias for :py:attr:`~get_row_names`, provided for back-compatibility."""
        return self.get_row_names()

    @rownames.setter
    def rownames(self, names: Optional[List[str]]):
        """Alias for :py:meth:`~set_row_names` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'row_names' is an in-place operation, use 'set_row_names' instead",
            UserWarning,
        )
        self.set_row_names(names, in_place=True)

    @property
    def row_names(self) -> Optional[ut.Names]:
        """Alias for :py:attr:`~get_row_names`, provided for back-compatibility."""
        return self.get_row_names()

    @row_names.setter
    def row_names(self, names: Optional[List[str]]):
        """Alias for :py:meth:`~set_row_names` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'row_names' is an in-place operation, use 'set_row_names' instead",
            UserWarning,
        )
        self.set_row_names(names, in_place=True)

    #############################
    ######>> column names <<#####
    #############################

    def get_column_names(self) -> Optional[ut.Names]:
        """
        Returns:
            List of column names, or None if no column names are available.
        """
        return self._column_names

    def set_column_names(
        self, names: Optional[List[str]], in_place: bool = False
    ) -> "BaseSE":
        """Set new column names.

        Args:
            names:
                New names, same as the number of columns.

                May be `None` to remove column names.

            in_place:
                Whether to modify the ``BaseSE`` in place.

        Returns:
            A modified ``BaseSE`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        if names is not None and not isinstance(names, ut.Names):
            names = ut.Names(names)

        _validate_cols(self._cols, names, self.shape)

        output = self._define_output(in_place)
        output._column_names = names
        return output

    @property
    def columnnames(self) -> Optional[ut.Names]:
        """Alias for :py:attr:`~get_column_names`, provided for back-compatibility."""
        return self.get_column_names()

    @columnnames.setter
    def columnnames(self, names: Optional[List[str]]):
        """Alias for :py:meth:`~set_column_names` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'column_names' is an in-place operation, use 'set_column_names' instead",
            UserWarning,
        )
        self.set_column_names(names, in_place=True)

    @property
    def colnames(self) -> Optional[ut.Names]:
        """Alias for :py:attr:`~get_column_names`, provided for back-compatibility."""
        return self.get_column_names()

    @colnames.setter
    def colnames(self, names: Optional[List[str]]):
        """Alias for :py:meth:`~set_column_names` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'column_names' is an in-place operation, use 'set_column_names' instead",
            UserWarning,
        )
        self.set_column_names(names, in_place=True)

    @property
    def col_names(self) -> Optional[ut.Names]:
        """Alias for :py:attr:`~get_column_names`, provided for back-compatibility."""
        return self.get_column_names()

    @col_names.setter
    def col_names(self, names: Optional[List[str]]):
        """Alias for :py:meth:`~set_column_names` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'column_names' is an in-place operation, use 'set_column_names' instead",
            UserWarning,
        )
        self.set_column_names(names, in_place=True)

    @property
    def column_names(self) -> Optional[ut.Names]:
        """Alias for :py:attr:`~get_column_names`, provided for back-compatibility."""
        return self.get_column_names()

    @column_names.setter
    def column_names(self, names: Optional[List[str]]):
        """Alias for :py:meth:`~set_column_names` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'column_names' is an in-place operation, use 'set_column_names' instead",
            UserWarning,
        )
        self.set_column_names(names, in_place=True)

    ###########################
    ######>> metadata <<#######
    ###########################

    def get_metadata(self) -> dict:
        """
        Returns:
            Dictionary of metadata for this object.
        """
        return self._metadata

    def set_metadata(self, metadata: dict, in_place: bool = False) -> "BaseSE":
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

    def set_assay_names(self, names: List[str], in_place: bool = False) -> "BaseSE":
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
            if assay < 0:
                raise IndexError("Index cannot be negative.")

            if assay > len(self.assay_names):
                raise IndexError("Index greater than the number of assays.")

            return self.assays[self.assay_names[assay]]
        elif isinstance(assay, str):
            if assay not in self.assays:
                raise AttributeError(f"Assay: {assay} does not exist.")

            return self.assays[assay]

        raise TypeError(
            f"'assay' must be a string or integer, provided '{type(assay)}'."
        )

    ##########################
    ######>> slicers <<#######
    ##########################

    def _normalize_row_slice(self, rows: Union[str, int, bool, Sequence]):
        _scalar = None
        if rows != slice(None):
            rows, _scalar = ut.normalize_subscript(
                rows, len(self._rows), self._row_names
            )

        return rows, _scalar

    def _normalize_column_slice(self, columns: Union[str, int, bool, Sequence]):
        _scalar = None
        if columns != slice(None):
            columns, _scalar = ut.normalize_subscript(
                columns, len(self._cols), self._column_names
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
        new_cols = self.column_data
        new_row_names = self._row_names
        new_col_names = self._column_names
        new_assays = {}

        if rows is None:
            rows = slice(None)

        if columns is None:
            columns = slice(None)

        if rows is not None:
            rows, _ = self._normalize_row_slice(rows=rows)
            new_rows = new_rows[rows, :]

            if new_row_names is not None:
                new_row_names = new_row_names[rows]

        if columns is not None and self.column_data is not None:
            columns, _ = self._normalize_column_slice(columns=columns)
            new_cols = new_cols[columns, :]

            if new_col_names is not None:
                new_col_names = new_col_names[columns]

        new_assays = self.subset_assays(rows=rows, columns=columns)

        return SliceResult(
            new_rows, new_cols, new_assays, new_row_names, new_col_names, rows, columns
        )

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
            column_data=slicer.columns,
            row_names=slicer.row_names,
            column_names=slicer.column_names,
            metadata=self._metadata,
        )

    def __getitem__(
        self,
        args: Union[int, str, Sequence, tuple],
    ) -> "BaseSE":
        """Subset a ``SummarizedExperiment``.

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
            ValueError:
                If too many or too few slices provided.

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

    ################################
    ######>> AnnData interop <<#####
    ################################

    def to_anndata(self):
        """Transform :py:class:`~BaseSE`-like into a :py:class:`~anndata.AnnData` representation.

        Returns:
            An ``AnnData`` representation of the experiment.
        """
        from anndata import AnnData

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()

        trows = self._rows.to_pandas()
        if self._row_names is not None:
            trows.index = self._row_names

        tcols = self._cols.to_pandas()
        if self._column_names is not None:
            tcols.index = self._column_names

        obj = AnnData(
            obs=tcols,
            var=trows,
            uns=self.metadata,
            layers=layers,
        )

        return obj
