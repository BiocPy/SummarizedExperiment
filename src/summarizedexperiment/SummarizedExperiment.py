from typing import Any, Dict, List, Optional
from warnings import warn

from biocframe import BiocFrame
from biocgenerics import (
    colnames,
    combine_cols,
    combine_rows,
    rownames,
    set_colnames,
    set_rownames,
)
from biocgenerics.combine import combine
from genomicranges import GenomicRanges
from numpy import empty

from .BaseSE import BaseSE
from .type_checks import is_list_of_subclass
from .types import SlicerArgTypes
from .utils.combiners import (
    combine_metadata,
)

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """Container to represent genomic experimental data (`assays`), features (`row_data`), sample data (`col_data`) and
    any other `metadata`.

    SummarizedExperiment follows the R/Bioconductor specification; rows are features, columns
    are samples.

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
        row_data: Optional[BiocFrame] = None,
        col_data: Optional[BiocFrame] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Initialize a Summarized Experiment (SE).

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

            metadata (Dict, optional): Additional experimental metadata describing the methods. Defaults to None.
        """

        if isinstance(row_data, GenomicRanges):
            warn(
                "`row_data` is `GenomicRanges`, consider using `RangeSummarizedExperiment`."
            )

        super().__init__(assays, row_data, col_data, metadata)

    def __getitem__(
        self,
        args: SlicerArgTypes,
    ) -> "SummarizedExperiment":
        """Subset a `SummarizedExperiment`.

        Args:
            args (SlicerArgTypes): Indices or names to slice. The tuple contains
                slices along dimensions (rows, cols).

                Each element in the tuple may be either an list of indices,
                boolean vector or :py:class:`~slice` object.

        Raises:
            ValueError: If too many or too few slices are provided.

        Returns:
            The same type as caller, with the sliced entries.
        """
        sliced_objs = self._slice(args)

        current_class_const = type(self)
        return current_class_const(
            assays=sliced_objs.assays,
            row_data=sliced_objs.row_data,
            col_data=sliced_objs.col_data,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        current_class_const = type(self)
        pattern = (
            f"Class {current_class_const.__name__} with {self.shape[0]} features and {self.shape[1]} "
            "samples \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  row_data: {self.row_data.columns if self.row_data is not None else None} \n"
            f"  col_data: {self.col_data.columns if self.col_data is not None else None}"
        )
        return pattern

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
                The resulting object will contain the same
                :py:attr:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.row_data` as the object its combined with.

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

    def combine_rows(
        self, *experiments: "BaseSE", fill_missing_assay: bool = False
    ) -> "BaseSE":
        """A more flexible version of ``rbind``.

        Permits differences in the number and identity of columns, differences in
        :py:attr:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.row_data` fields, and even differences
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
            The resulting object will contain the same
            :py:attr:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment.col_data` as the object its combined with.
        """

        if not is_list_of_subclass(experiments, BaseSE):
            raise TypeError(
                "Not all provided objects are `SummarizedExperiment`-like objects."
            )

        ses = [self] + list(experiments)
        new_metadata = combine_metadata(ses)

        _all_row_data = [getattr(e, "row_data") for e in ses]
        new_coldata = combine_rows(*_all_row_data)

        # _all_row_data = [getattr(e, "row_data") for e in ses]
        # new_rowdata = combine(*_all_row_data)
        new_rowdata = self.col_data

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
                        new_assays[aname] = combine_rows(
                            new_assays[aname], empty(shape=obj.shape)
                        )
                    else:
                        raise AttributeError(
                            f"Assay: `{aname}` does not exist in all experiments."
                        )
                else:
                    new_assays[aname] = combine_rows(
                        new_assays[aname], obj.assays[aname]
                    )

        current_class_const = type(self)
        return current_class_const(new_assays, new_rowdata, new_coldata, new_metadata)


@rownames.register(SummarizedExperiment)
def _rownames_se(x: SummarizedExperiment):
    return rownames(x.row_data)


@set_rownames.register(SummarizedExperiment)
def _set_rownames_se(x: Any, names: List[str]):
    set_rownames(x.row_data, names)


@colnames.register(SummarizedExperiment)
def _colnames_se(x: SummarizedExperiment):
    return rownames(x.col_data)


@set_colnames.register(SummarizedExperiment)
def _set_colnames_se(x: Any, names: List[str]):
    set_rownames(x.col_data, names)


@combine.register(SummarizedExperiment)
def _combine_se(x: Any):
    if not isinstance(x[0], SummarizedExperiment):
        raise TypeError("First element is not a summarized experiment!")

    return x[0].combine_rows(x[1])


@combine_rows.register(SummarizedExperiment)
def _combine_rows_se(x: Any):
    if not isinstance(x[0], SummarizedExperiment):
        raise TypeError("First element is not a summarized experiment!")

    return x[0].combine_rows(x[1])


@combine_cols.register(SummarizedExperiment)
def _combine_cols_se(x: Any):
    if not isinstance(x[0], SummarizedExperiment):
        raise TypeError("First element is not a summarized experiment!")

    return x[0].combine_cols(x[1])
