from typing import Any, Dict, List, Optional
from warnings import warn

import biocframe
import biocutils as ut
from genomicranges import GenomicRanges

from ._combineutils import (
    check_assays_are_equal,
    merge_assays,
    merge_se_colnames,
    merge_se_rownames,
    relaxed_merge_assays,
)
from .BaseSE import BaseSE

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment(BaseSE):
    """Container to represents genomic experiment data (`assays`), features (`row_data`), sample data (`column_data`)
    and any other `metadata`.

    SummarizedExperiment follows the R/Bioconductor specification; rows are features, columns are samples.
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
        """Initialize a Summarized Experiment (SE).

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

            row_data:
                Features, must be the same length as the number of rows of
                the matrices in assays.

                Feature information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            column_data:
                Sample data, must be the same length as the number of
                columns of the matrices in assays.

                Sample information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            row_names:
                A list of strings, same as the number of rows.

                If ``row_names`` are not provided, these are inferred from
                ``row_data``.

                Defaults to None.

            column_names:
                A list of string, same as the number of columns.

                if ``column_names`` are not provided, these are inferred from
                ``column_data``.

                Defaults to None.

            metadata:
                Additional experimental metadata describing the methods.
                Defaults to None.

            validate:
                Internal use only.
        """

        if isinstance(row_data, GenomicRanges):
            warn(
                "`row_data` is `GenomicRanges`, consider using `RangeSummarizedExperiment`."
            )

        super().__init__(
            assays,
            row_data=row_data,
            column_data=column_data,
            row_names=row_names,
            column_names=column_names,
            metadata=metadata,
            validate=validate,
        )


############################
######>> combine ops <<#####
############################


@ut.combine_rows.register(SummarizedExperiment)
def combine_rows(*x: SummarizedExperiment) -> SummarizedExperiment:
    """Combine multiple ``SummarizedExperiment`` objects by row.

    All assays must contain the same assay names. If you need a
    flexible combine operation, checkout :py:func:`~relaxed_combine_rows`.

    Returns:
        A combined ``SummarizedExperiment``.
    """
    first = x[0]
    _all_assays = [y.assays for y in x]
    check_assays_are_equal(_all_assays)
    _new_assays = merge_assays(_all_assays, by="row")

    _all_rows = [y._rows for y in x]
    _new_rows = ut.combine_rows(*_all_rows)
    _new_row_names = merge_se_rownames(x)

    current_class_const = type(first)
    return current_class_const(
        assays=_new_assays,
        row_data=_new_rows,
        column_data=first._cols,
        row_names=_new_row_names,
        column_names=first._column_names,
        metadata=first._metadata,
    )


@ut.combine_columns.register(SummarizedExperiment)
def combine_columns(*x: SummarizedExperiment) -> SummarizedExperiment:
    """Combine multiple ``SummarizedExperiment`` objects by column.

    All assays must contain the same assay names. If you need a
    flexible combine operation, checkout :py:func:`~relaxed_combine_columns`.

    Returns:
        A combined ``SummarizedExperiment``.
    """
    first = x[0]
    _all_assays = [y.assays for y in x]
    check_assays_are_equal(_all_assays)
    _new_assays = merge_assays(_all_assays, by="column")

    _all_cols = [y._cols for y in x]
    _new_cols = ut.combine_rows(*_all_cols)
    _new_col_names = merge_se_colnames(x)

    current_class_const = type(first)
    return current_class_const(
        assays=_new_assays,
        row_data=first._rows,
        column_data=_new_cols,
        row_names=first._row_names,
        column_names=_new_col_names,
        metadata=first._metadata,
    )


@ut.relaxed_combine_rows.register(SummarizedExperiment)
def relaxed_combine_rows(*x: SummarizedExperiment) -> SummarizedExperiment:
    """A relaxed version of the :py:func:`~biocutils.combine_rows.combine_rows` method for
    :py:class:`~SummarizedExperiment` objects.  Whereas ``combine_rows`` expects that all objects have the same columns,
    ``relaxed_combine_rows`` allows for different columns. Absent columns in any object are filled in with appropriate
    placeholder values before combining.

    Args:
        x:
            One or more ``SummarizedExperiment`` objects, possibly with differences in the
            number and identity of their columns.

    Returns:
        A ``SummarizedExperiment`` that combines all ``experiments`` along their rows and contains
        the union of all columns. Columns absent in any ``x`` are filled in
        with placeholders consisting of Nones or masked NumPy values.
    """
    first = x[0]
    _new_assays = relaxed_merge_assays(x, by="row")

    _all_rows = [y._rows for y in x]
    _new_rows = biocframe.relaxed_combine_rows(*_all_rows)
    _new_row_names = merge_se_rownames(x)

    current_class_const = type(first)
    return current_class_const(
        assays=_new_assays,
        row_data=_new_rows,
        column_data=first._cols,
        row_names=_new_row_names,
        column_names=first._column_names,
        metadata=first._metadata,
    )


@ut.relaxed_combine_columns.register(SummarizedExperiment)
def relaxed_combine_columns(*x: SummarizedExperiment) -> SummarizedExperiment:
    """A relaxed version of the :py:func:`~biocutils.combine_rows.combine_columns` method for
    :py:class:`~SummarizedExperiment` objects.  Whereas ``combine_columns`` expects that all objects have the same rows,
    ``relaxed_combine_columns`` allows for different rows. Absent columns in any object are filled in with appropriate
    placeholder values before combining.

    Args:
        x:
            One or more ``SummarizedExperiment`` objects, possibly with differences in the
            number and identity of their rows.

    Returns:
        A ``SummarizedExperiment`` that combines all ``experiments`` along their columns and contains
        the union of all rows. Rows absent in any ``x`` are filled in
        with placeholders consisting of Nones or masked NumPy values.
    """
    first = x[0]
    _new_assays = relaxed_merge_assays(x, by="column")

    _all_cols = [y._cols for y in x]
    _new_cols = biocframe.relaxed_combine_rows(*_all_cols)
    _new_col_names = merge_se_colnames(x)

    current_class_const = type(first)
    return current_class_const(
        assays=_new_assays,
        row_data=first._rows,
        column_data=_new_cols,
        row_names=first._row_names,
        column_names=_new_col_names,
        metadata=first._metadata,
    )


@ut.extract_row_names.register(SummarizedExperiment)
def _rownames_se(x: SummarizedExperiment):
    return x.get_row_names()


@ut.extract_column_names.register(SummarizedExperiment)
def _colnames_se(x: SummarizedExperiment):
    return x.get_column_names()
