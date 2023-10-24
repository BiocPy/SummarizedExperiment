from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

from biocframe import BiocFrame
from biocgenerics import (
    colnames,
    combine_cols,
    rownames,
    set_colnames,
    set_rownames,
)
from biocgenerics import combine_rows
from biocutils import is_list_of_type
from genomicranges import GenomicRanges
from numpy import empty

from ._frameutils import _sanitize_frame
from .type_checks import (
    is_bioc_or_pandas_frame,
    is_list_of_subclass,
    is_matrix_like,
)
from .types import (
    MatrixSlicerTypes,
    SlicerArgTypes,
    SlicerResult,
)
from .utils.combiners import (
    combine_metadata,
)
from .utils.slicer import get_indexes_from_bools, get_indexes_from_names

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche"
__license__ = "MIT"


class BaseSE:
    """Base class for `SummarizedExperiment`. This class provides common properties and methods that can be utilized
    across all derived classes.

    This container represents genomic experiment data in the form of ``assays``,
    features in ``row_data``, sample data in ``col_data``, and any other relevant ``metadata``.

    Attributes:
        assays (Dict[str, Any]): A dictionary containing matrices, with assay names as keys
            and 2-dimensional matrices represented as either
            :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

            Alternatively, you may use any 2-dimensional matrix that has the ``shape`` property and
            implements the slice operation using the ``__getitem__`` dunder method.

            All matrices in assays must be 2-dimensional and have the same shape
            (number of rows, number of columns).

        row_data (BiocFrame, optional): Features, must be the same length as the number of rows of
            the matrices in assays. Feature information is coerced to a
            :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

        col_data (BiocFrame, optional): Sample data, must be the same length as the number of
            columns of the matrices in assays. Sample information is coerced to a
            :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

        metadata (Dict, optional): Additional experimental metadata describing the methods. Defaults to None.
    """

    def __init__(
        self,
        assays: Dict[str, Any],
        rows: Optional[BiocFrame] = None,
        cols: Optional[BiocFrame] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Initialize an instance of `BaseSE`.

        Args:
            assays (Dict[str, Any]): A dictionary containing matrices, with assay names as keys
                and 2-dimensional matrices represented as either
                :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

                Alternatively, you may use any 2-dimensional matrix that has the ``shape`` property and
                implements the slice operation using the ``__getitem__`` dunder method.

                All matrices in assays must be 2-dimensional and have the same shape
                (number of rows, number of columns).

            row_data (BiocFrame, optional): Features, must be the same length as the number of rows of
                the matrices in assays. Feature information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            col_data (BiocFrame, optional): Sample data, must be the same length as the number of
                columns of the matrices in assays. Sample information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            metadata (dict, optional): Additional experimental metadata describing the methods. Defaults to None.
        """

        self._shape = None
        self._rows = None
        self._cols = None

        if assays is None or not isinstance(assays, dict) or len(assays.keys()) == 0:
            raise Exception(
                "`assays` must be a dictionary and contain atleast one 2-dimensional matrix."
            )

        self._validate_assays(assays)
        self._assays = assays

        # should have _shape by now
        if self._shape is None:
            raise RuntimeError("Cannot extract shape from assays!")

        self._set_rows(rows)
        self._set_cols(cols)

        self._metadata = metadata

    def _set_rows(self, rows: Optional[BiocFrame]):
        rows = _sanitize_frame(rows, self._shape[0])
        self._validate_rows(rows)
        self._rows = rows

    def _set_cols(self, cols: Optional[BiocFrame]):
        cols = _sanitize_frame(cols, self._shape[1])
        self._validate_cols(cols)
        self._cols = cols

    def _validate(self):
        """Internal wrapper method to validate the object."""
        # validate assays to make sure they are have same dimensions
        self._validate_assays(self._assays)
        self._validate_rows(self._rows)
        self._validate_cols(self._cols)

    def _validate_assays(
        self,
        assays: Dict[str, Any],
    ):
        """Internal method to validate experiment data (assays).

        Args:
            assays (Dict[str, Any]): Experiment data.

        Raises:
            ValueError:
                If ``assays`` contain more than 2 dimensions.
                If all ``assays`` do not have the same dimensions.
            TypeError: If ``assays`` contains an unsupported matrix representation.
        """

        for asy, mat in assays.items():
            if not is_matrix_like(mat):
                raise TypeError(f"Assay: '{asy}' is not a supported matrix type.")

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
                    f"dimensions mismatch, '{asy}' must be of shape '{self._shape}'"
                    f" but provided '{mat.shape}'."
                )

    def _validate_rows(self, rows: BiocFrame):
        """Internal method to validate feature information (row_data).

        Args:
            rows (BiocFrame): Features to validate.

        Raises:
            ValueError: When the number of rows in ``rows`` does not match the number of rows across assays.
            TypeError: When ``rows`` is not a :py:class:`~biocframe.BiocFrame.BiocFrame`.
        """
        if not is_bioc_or_pandas_frame(rows):
            raise TypeError(
                "`row_data` must be either a pandas `DataFrame` or a `BiocFrame`"
                f" object, provided {type(rows)}."
            )

        if rows.shape[0] != self._shape[0]:
            raise ValueError(
                f"Number of features mismatch with number of rows in assays. Must be '{self._shape[0]}'"
                f" but provided '{rows.shape[0]}'."
            )

    def _validate_cols(self, cols: BiocFrame):
        """Internal method to validate sample information (col_data).

        Args:
            cols (BiocFrame): Sample information (col_data).

        Raises:
            ValueError: When the number of columns in ``cols`` does not match the number of columns across assays.
            TypeError: When ``cols`` is not a :py:class:`~biocframe.BiocFrame.BiocFrame`.
        """
        if not is_bioc_or_pandas_frame(cols):
            raise TypeError(
                "`col_data` must be either a pandas `DataFrame` or a `BiocFrame`"
                f" object, provided {type(cols)}."
            )

        if cols.shape[0] != self._shape[1]:
            raise ValueError(
                f"Number of samples mismatch with number of columns in assays. Must be '{self._shape[1]}'"
                f" but provided '{cols.shape[0]}'."
            )

    @property
    def assays(
        self,
    ) -> Dict[str, Any]:
        """Retrieve all experiment data (assays).

        Returns:
            Dict[str, Any]: A dictionary with experiment names as keys,
            and matrices as corresponding values.
        """
        return self._assays

    @assays.setter
    def assays(
        self,
        assays: Dict[str, Any],
    ):
        """Set new experiment data (assays).

        Args:
            assays (Dict[str, Any]): New assays.
        """
        self._validate_assays(assays)
        self._assays = assays

    @property
    def row_data(self) -> BiocFrame:
        """Get features.

        Returns:
            BiocFrame: Feature information.
        """
        return self._rows

    @row_data.setter
    def row_data(self, rows: Optional[BiocFrame]):
        """Set features.

        Args:
            rows (BiocFrame, optional): New feature information.
                If ``rows`` is None, an empty :py:class:`~biocframe.BiocFrame.BiocFrame`
                object is created.
        """
        self._set_rows(rows)

    @property
    def col_data(self) -> BiocFrame:
        """Get sample data.

        Returns:
            BiocFrame: Sample information.
        """
        return self._cols

    @col_data.setter
    def col_data(self, cols: Optional[BiocFrame]):
        """Set sample data.

        Args:
            cols (BiocFrame, optional): New sample data.
                If ``cols`` is None, an empty :py:class:`~biocframe.BiocFrame.BiocFrame`
                object is created.
        """
        self._set_cols(cols)

    @property
    def metadata(self) -> Optional[dict]:
        """Retrieve metadata.

        Returns:
            Optional[dict]: A metadata object, typically in the form of a dictionary.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[dict]):
        """Set metadata.

        Args:
            metadata (Optional[dict]): New metadata object.
        """
        if metadata is None:
            metadata = {}

        self._metadata = metadata

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the experiment.

        Returns:
            Tuple[int, int]: A tuple (m,n),
            where `m` is the number of features/rows, and `n` is the number of samples/columns.
        """
        return self._shape

    @property
    def dims(self) -> Tuple[int, int]:
        """Dimensions of the experiment.

        Alias to :py:attr:`~summarizedexperiment.BaseSE.BaseSE.shape`.

        Returns:
            Tuple[int, int]: A tuple (m,n),
            where `m` is the number of features/rows, and `n` is the number of samples/columns.
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
    def assay_names(self, names: List[str]):
        """Replace all :py:attr:`~summarizedexperiment.BaseSE.BaseSE.assays`'s names.

        Args:
            names (List[str]): New names.

        Raises:
            ValueError: If length of ``names`` does not match the number of assays.
        """
        current_names = self.assay_names
        if len(names) != len(current_names):
            raise ValueError("Length of `names` does not match the number of `assays`.")

        new_assays = OrderedDict()
        for idx in range(len(names)):
            new_assays[names[idx]] = self._assays.pop(current_names[idx])

        self._assays = new_assays

    def __repr__(self) -> str:
        current_class_const = type(self)
        pattern = (
            f"Class {current_class_const.__name__} with {self.shape[0]} features and {self.shape[1]} samples \n"
            f"  assays: {', '.join(list(self.assays.keys()))} \n"
            f"  features: {self.row_data.columns if self.row_data is not None else None} \n"
            f"  sample data: {self.col_data.columns if self.col_data is not None else None}"
        )
        return pattern

    def assay(self, index_or_name: Union[int, str]) -> Any:
        """Convenience method to access an :py:attr:`~summarizedexperiment.BaseSE.BaseSE.assays` by name or index.

        Args:
            name (Union[int, str]): Name or index position of the assay.

        Raises:
            AttributeError: If the assay name does not exist.
            IndexError: If index is greater than the number of assays.

        Returns:
            Any: Experiment data.
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
    ) -> Dict[str, Any]:
        """Subset all assays using a slice defined by rows and columns.

        If both ``row_indices`` and ``col_indices`` are None, a copy of the
        current assays is returned.

        Args:
            row_indices (MatrixSlicerTypes, optional): Row indices to subset.

                ``row_indices`` may be a list of integer indices to subset.

                Alternatively ``row_indices`` may be a boolean vector specifying
                `True` to keep the row or `False` to remove. The length of the boolean
                vector must match the number of rows in the experiment.

                Alternatively, ``row_indices`` may be a :py:class:`~slice` object.

                Defaults to None.

            col_indices (MatrixSlicerTypes, optional): Column indices to subset.

                ``col_indices`` may be a list of integer indices to subset.

                Alternatively ``col_indices`` may be a boolean vector specifying
                `True` to keep the column or `False` to remove. The length of the boolean
                vector must match the number of columns in the experiment.

                Alternatively, ``col_indices`` may be a :py:class:`~slice` object.

                Defaults to None.

        Raises:
            warning: If ``row_indices`` and ``col_indices`` are both None.

        Returns:
            Dict[str, Any]: Sliced experimental data.
        """

        if row_indices is None and col_indices is None:
            warn("No slice is provided, this returns a copy of all assays!")
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
        """Internal method to perform slicing.

        Args:
            args (SlicerArgTypes): Indices or names to slice. The tuple contains
                slices along dimensions (rows, cols).

                Each element in the tuple may be either an list of indices,
                boolean vector or :py:class:`~slice` object.

                Defaults to None.

        Raises:
            ValueError: If too many or too few slices are provided.

        Returns:
            SlicerResult: The sliced tuple.
        """

        if isinstance(args, tuple):
            if len(args) == 0:
                raise ValueError("Must contain atleast one slice.")

            row_indices = args[0]
            col_indices = None

            if len(args) > 1:
                col_indices = args[1]
            elif len(args) > 2:
                raise ValueError("Arguments contain too many slices.")
        elif isinstance(args, list) or isinstance(args, slice):
            row_indices = args
            col_indices = None
        else:
            raise ValueError("Arguments are not supported.")

        new_rows = self.row_data
        new_cols = self.col_data
        new_assays = None

        if row_indices is not None and self.row_data is not None:
            if is_list_of_type(row_indices, str):
                row_indices = get_indexes_from_names(
                    rownames(self.row_data), row_indices
                )
            elif is_list_of_type(row_indices, bool):
                if len(row_indices) != self.shape[0]:
                    raise ValueError(
                        "`row_indices` is a boolean vector, length of vector must match the ",
                        "number of rows.",
                    )
                row_indices = get_indexes_from_bools(row_indices)

            if is_list_of_type(row_indices, int) or isinstance(row_indices, slice):
                new_rows = new_rows[row_indices, :]
            else:
                raise TypeError("Arguments to slice rows is not supported!")

        if col_indices is not None and self.col_data is not None:
            if is_list_of_type(col_indices, str):
                col_indices = get_indexes_from_names(
                    rownames(self.col_data), col_indices
                )
            elif is_list_of_type(col_indices, bool):
                if len(col_indices) != self.shape[1]:
                    raise ValueError(
                        "`col_indices` is a boolean vector, length of vector must match the ",
                        "number of columns.",
                    )
                col_indices = get_indexes_from_bools(col_indices)

            if is_list_of_type(col_indices, int) or isinstance(col_indices, slice):
                new_cols = new_cols[col_indices, :]
            else:
                raise TypeError("Arguments to slice column is not supported!")

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
        return rownames(self.row_data)

    @row_names.setter
    def row_names(self, names: List[str]):
        """Set row/feature names for the experiment.

        Args:
            names (List[str]): New feature names.

        Raises:
            ValueError: If length of ``names`` is not same as the number of rows.
        """
        if len(names) != self.shape[0]:
            raise ValueError("Length of `names` must be the same as number of rows.")

        set_rownames(self.row_data, names)

    @property
    def colnames(self) -> List[str]:
        """Get column/sample names.

        Returns:
            List[str]: List of sample names.
        """
        return rownames(self.col_data)

    @colnames.setter
    def colnames(self, names: List[str]):
        """Set column/sample names for the experiment.

        Args:
            names (List[str]): New samples names.

        Raises:
            ValueError: If length of ``names`` is not same as the number of rows.
        """
        if len(names) != self.shape[1]:
            raise ValueError("Length of `names` must be the same as number of columns.")

        set_rownames(self.col_data, names)

    def to_anndata(self):
        """Coerce :py:class:`summarizedexperiment.BaseSE`-like into an :py:class:`~anndata.AnnData` representation.

        Returns:
            AnnData: An `AnnData` representation of the experiment.
        """
        from anndata import AnnData

        layers = OrderedDict()
        for asy, mat in self.assays.items():
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
        self, *experiments: "BaseSE", fill_missing_assay: bool = False
    ) -> "BaseSE":
        """A more flexible version of ``cbind``.

        Permits differences in the number and identity of rows, differences in
        :py:attr:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.col_data` fields, and even differences
        in the available `assays` among :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`-derived objects
        being combined.

        Args:
            *experiments (BaseSE): `SummarizedExperiment`-like objects to concatenate.

            fill_missing_assay (bool): Fills missing assays across experiments with an empty sparse matrix.

        Raises:
            TypeError:
                If any of the provided objects are not "SummarizedExperiment"-like objects.

        Returns:
            Same type as the caller with the combined experiments.
        """

        if not is_list_of_subclass(experiments, BaseSE):
            raise TypeError(
                "Not all provided objects are `SummarizedExperiment`-like objects."
            )

        ses = [self] + list(experiments)
        new_metadata = combine_metadata(ses)

        _all_col_data = [getattr(e, "col_data") for e in ses]
        new_coldata = combine_rows(*_all_col_data)

        # _all_row_data = [getattr(e, "row_data") for e in ses]
        # new_rowdata = combine(*_all_row_data)
        new_rowdata = self.row_data

        new_assays = self.assays.copy()
        unique_assay_names = {assay_name for se in ses for assay_name in se.assay_names}

        for aname in unique_assay_names:
            if aname not in self.assays:
                if fill_missing_assay is True:
                    new_assays[aname] = empty(shape=self.shape)
                else:
                    raise AttributeError(
                        f"Assay: `{aname}` does not exist in all experiments."
                    )

        for obj in experiments:
            for aname in unique_assay_names:
                if aname not in obj.assays:
                    if fill_missing_assay is True:
                        new_assays[aname] = combine_cols(
                            new_assays[aname], empty(shape=obj.shape)
                        )
                    else:
                        raise AttributeError(
                            f"Assay: `{aname}` does not exist in all experiments."
                        )
                else:
                    new_assays[aname] = combine_cols(
                        new_assays[aname], obj.assays[aname]
                    )

        current_class_const = type(self)
        return current_class_const(new_assays, new_rowdata, new_coldata, new_metadata)


@rownames.register(BaseSE)
def _rownames_se(x: BaseSE):
    return rownames(x.row_data)


@set_rownames.register(BaseSE)
def _set_rownames_se(x: Any, names: List[str]):
    set_rownames(x.row_data, names)


@colnames.register(BaseSE)
def _colnames_se(x: BaseSE):
    return rownames(x.col_data)


@set_colnames.register(BaseSE)
def _set_colnames_se(x: Any, names: List[str]):
    set_rownames(x.col_data, names)
