from typing import Dict, List, Literal, Optional, Sequence, Union
from warnings import warn

import numpy as np
from biocframe import BiocFrame
from genomicranges import GenomicRanges, GenomicRangesList, SeqInfo

from ._frameutils import is_pandas
from .SummarizedExperiment import SummarizedExperiment
from .types import MatrixTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

GRangesOrGRangesList = Union[GenomicRanges, GenomicRangesList]
GRangesOrRangeSE = Union[GRangesOrGRangesList, "RangedSummarizedExperiment"]


# TODO: technically should be in _type_checks but causes circular imports.
def _check_gr_or_rse(x: GRangesOrRangeSE):
    """Check if ``x`` is either a `RangedSummarizedExperiment` or `GenomicRanges`.

    Args:
        x:
            Object to check.

    Raises:
        TypeError:
            Object is not a `RangedSummarizedExperiment` or `GenomicRanges`.
    """
    if not isinstance(
        x, (RangedSummarizedExperiment, GenomicRanges, GenomicRangesList)
    ):
        raise TypeError(
            "'x' is not a `RangedSummarizedExperiment`, `GenomicRanges` or `GenomicRangesList`."
        )


def _access_granges(x: GRangesOrRangeSE) -> GenomicRanges:
    """Access ranges from the object.

    Args:
        x:
            Input object.

    Returns:
        `GenomicRanges` object.
    """
    qranges = x
    if isinstance(x, RangedSummarizedExperiment):
        qranges = x.row_ranges

    return qranges


def _validate_rowranges(row_ranges, shape):
    if not (isinstance(row_ranges, (GenomicRanges, GenomicRangesList))):
        raise TypeError(
            "`row_ranges` must be a `GenomicRanges` or `GenomicRangesList`"
            f" , provided {type(row_ranges)}."
        )

    if len(row_ranges) != shape[0]:
        raise ValueError(
            "Number of features in `row_ranges` and number of rows in assays do not match."
        )


def _sanitize_ranges_frame(frame, num_rows: int):
    frame = frame if frame is not None else GenomicRangesList.empty(n=num_rows)

    if is_pandas(frame):
        frame = GenomicRanges.from_pandas(frame)

    return frame


class RangedSummarizedExperiment(SummarizedExperiment):
    """RangedSummarizedExperiment class to represent genomic experiment data, genomic features as
    :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or
    :py:class:`~genomicranges.GenomicRangesList.GenomicRangesList` sample data and any additional experimental metadata.

    Note: If ``row_ranges`` is empty, None or not a
    :py:class:`genomicranges.GenomicRanges.GenomicRanges` object, use a
    :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` instead.
    """

    def __init__(
        self,
        assays: Dict[str, MatrixTypes],
        row_ranges: Optional[GRangesOrGRangesList] = None,
        row_data: Optional[BiocFrame] = None,
        col_data: Optional[BiocFrame] = None,
        metadata: Optional[dict] = None,
        validate: bool = True,
    ) -> None:
        """Initialize a `RangedSummarizedExperiment` (RSE) object.

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

            row_ranges:
                Genomic features, must be the same length as the number of rows of
                the matrices in assays.

            row_data:
                Features, must be the same length as the number of rows of
                the matrices in assays.

                Feature information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            col_data:
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
        super().__init__(
            assays,
            row_data=row_data,
            col_data=col_data,
            metadata=metadata,
            validate=validate,
        )

        self._row_ranges = _sanitize_ranges_frame(row_ranges, self._shape[0])
        if validate:
            _validate_rowranges(self._row_ranges, self._shape)

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``RangedSummarizedExperiment``.
        """
        from copy import deepcopy

        _assays_copy = deepcopy(self._assays)
        _rows_copy = deepcopy(self._rows)
        _rowranges_copy = deepcopy(self._row_ranges)
        _cols_copy = deepcopy(self._cols)
        _metadata_copy = deepcopy(self.metadata)

        current_class_const = type(self)
        return current_class_const(
            assays=_assays_copy,
            row_ranges=_rowranges_copy,
            row_data=_rows_copy,
            col_data=_cols_copy,
            metadata=_metadata_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``RangedSummarizedExperiment``.
        """
        current_class_const = type(self)
        return current_class_const(
            assays=self._assays,
            row_ranges=self._row_ranges,
            row_data=self._rows,
            col_data=self._cols,
            metadata=self._metadata,
        )

    def copy(self):
        """Alias for :py:meth:`~__copy__`."""
        return self.__copy__()

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self) -> str:
        pattern = (
            f"Class  {type(self).__name__} with {self.shape[0]} features and {self.shape[1]} "
            "samples \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  row_data: {self._rows.columns if self._rows is not None else None} \n"
            f"  col_data: {self._cols.columns if self._cols is not None else None}"
        )
        return pattern

    ############################
    ######>> row_ranges <<######
    ############################

    def get_rowranges(self) -> GRangesOrGRangesList:
        """Get genomic feature information.

        Returns:
            Genomic feature information.
        """
        return self._row_ranges

    def set_rowranges(
        self, row_ranges: Optional[GRangesOrGRangesList], in_place: bool = False
    ):
        """Set new genomic features.

        Args:
            row_ranges:
                Genomic features, must be the same length as the number of rows of
                the matrices in assays.

            in_place:
                Whether to modify the ``RangeSummarizedExperiment`` in place.

        Returns:
            A modified ``RangeSummarizedExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        rows = _sanitize_ranges_frame(row_ranges, self._shape[0])
        _validate_rowranges(rows, self._shape)

        output = self._define_output(in_place)
        output._rows = rows
        return output

    @property
    def row_ranges(self) -> GRangesOrGRangesList:
        """Alias for :py:meth:`~get_rowranges`."""
        return self.get_rowranges()

    @row_ranges.setter
    def row_ranges(self, row_ranges: GRangesOrGRangesList) -> None:
        """Alias for :py:meth:`~set_rowranges`."""
        warn(
            "Setting property 'row_ranges' is an in-place operation, use 'set_rowranges' instead",
            UserWarning,
        )
        return self.set_rowranges(row_ranges=row_ranges, in_place=True)

    #################################
    ######>> range accessors <<######
    #################################

    @property
    def end(self) -> np.ndarray:
        """Get genomic end positions for each feature or row in experimental data.

        Returns:
            A :py:class:`numpy.ndarray` of end positions.
        """
        return self.row_ranges.end

    @property
    def start(self) -> np.ndarray:
        """Get genomic start positions for each feature or row in experimental data.

        Returns:
            A :py:class:`numpy.ndarray` of start positions.
        """
        return self.row_ranges.start

    @property
    def seqnames(self) -> List[str]:
        """Get sequence or chromosome names.

        Returns:
            List of all chromosome names.
        """
        return self.row_ranges.seqnames

    @property
    def strand(self) -> np.ndarray:
        """Get strand information.

        Returns:
            A :py:class:`numpy.ndarray` of strand information.
        """
        return self.row_ranges.strand

    @property
    def width(self) -> np.ndarray:
        """Get widths of ``row_ranges``.

        Returns:
            A :py:class:`numpy.ndarray` of widths for each interval.
        """
        return self.row_ranges.width

    @property
    def seq_info(self) -> SeqInfo:
        """Get sequence information object (if available).

        Returns:
            Sequence information.
        """
        return self.row_ranges.seq_info

    ##########################
    ######>> slicers <<#######
    ##########################

    def __getitem__(
        self,
        args: Union[int, str, Sequence, tuple],
    ) -> "RangedSummarizedExperiment":
        """Subset a `RangedSummarizedExperiment`.

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
                If too many or too few slices are provided.

        Returns:
            Sliced `RangedSummarizedExperiment` object.
        """
        sliced_objs = self._generic_slice(args)

        new_row_ranges = None
        if sliced_objs.row_indices != slice(None):
            new_row_ranges = self.row_ranges[sliced_objs.row_indices]

        current_class_const = type(self)
        return current_class_const(
            assays=sliced_objs.assays,
            row_ranges=new_row_ranges,
            row_data=sliced_objs.row_data,
            col_data=sliced_objs.col_data,
            metadata=self.metadata,
        )

    ############################
    ######>> range ops <<#######
    ############################

    def coverage(
        self, shift: int = 0, width: Optional[int] = None, weight: int = 1
    ) -> Dict[str, np.ndarray]:
        """Calculate coverage for each chromosome.

        Args:
            shift:
                Shift all genomic positions. Defaults to 0.

            width:
                Restrict the width of all
                chromosomes. Defaults to None.

            weight:
                Weight to use. Defaults to 1.

        Returns:
            A dictionary with chromosome names as keys and the
            coverage vector as value.
        """
        return self.row_ranges.coverage(shift=shift, width=width, weight=weight)

    def nearest(
        self,
        query: GRangesOrRangeSE,
        select: Literal["all", "arbitrary"] = "all",
        ignore_strand: bool = False,
    ) -> Optional[List[Optional[int]]]:
        """Search nearest positions both upstream and downstream that overlap with each range in ``query``.

        Args:
            query:
                Query intervals to find nearest positions.

                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.

            select:
                Determine what hit to choose when there are
                multiple hits for an interval in ``query``.

            ignore_strand:
                Whether to ignore strand. Defaults to False.

        Raises:
            If ``query`` is not a ``RangedSummarizedExperiment``
                or ``GenomicRanges``.

        Returns:
            A list with the same length as ``query``,
            containing hits to nearest indices.
        """

        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.row_ranges.nearest(
            query=qranges, select=select, ignore_strand=ignore_strand
        )
        return res

    def precede(
        self,
        query: GRangesOrRangeSE,
        select: Literal["all", "arbitrary"] = "all",
        ignore_strand: bool = False,
    ) -> Optional[List[Optional[int]]]:
        """Search nearest positions only downstream that overlap with each range in ``query``.

        Args:
            query:
                Query intervals to find nearest positions.

                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.

            select:
                Determine what hit to choose when there are
                multiple hits for an interval in ``query``.

            ignore_strand:
                Whether to ignore strand. Defaults to False.

        Raises:
            If ``query`` is not a ``RangedSummarizedExperiment`` or
            ``GenomicRanges``.

        Returns:
            A List with the same length as ``query``,
            containing hits to nearest indices.
        """

        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.row_ranges.precede(
            query=qranges, select=select, ignore_strand=ignore_strand
        )
        return res

    def follow(
        self,
        query: GRangesOrRangeSE,
        select: Literal["all", "arbitrary"] = "all",
        ignore_strand: bool = False,
    ) -> Optional[List[Optional[int]]]:
        """Search nearest positions only upstream that overlap with each range in ``query``.

        Args:
            query:
                Query intervals to find nearest positions.

                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.

            select:
                Determine what hit to choose when there are
                multiple hits for an interval in ``query``.

            ignore_strand:
                Whether to ignore strand. Defaults to False.

        Raises:
            If ``query`` is not a ``RangedSummarizedExperiment`` or
            ``GenomicRanges``.

        Returns:
            A List with the same length as ``query``,
            containing hits to nearest indices.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.row_ranges.follow(
            query=qranges, select=select, ignore_strand=ignore_strand
        )
        return res

    def flank(
        self,
        width: int,
        start: bool = True,
        both: bool = False,
        ignore_strand: bool = False,
        in_place: bool = False,
    ) -> "RangedSummarizedExperiment":
        """Compute flanking ranges for each range.

        Refer to either :py:meth:`~genomicranges.GenomicRanges.GenomicRanges.flank` or the
        Bioconductor documentation for more details.

        Args:
            width:
                Width to flank by. May be negative.

            start:
                Whether to only flank starts. Defaults to True.

            both:
                Whether to flank both starts and ends. Defaults to False.

            ignore_strand:
                Whether to ignore strands. Defaults to False.

            in_place:
                Whether to modify the ``GenomicRanges`` object in place.

        Returns:
            A new `RangedSummarizedExperiment` object with the flanked ranges,
            either as a copy of the original or as a reference to the
            (in-place-modified) original.
        """
        new_ranges = self.row_ranges.flank(
            width=width,
            start=start,
            both=both,
            ignore_strand=ignore_strand,
            in_place=False,
        )

        output = self._define_output(in_place=in_place)
        output._row_ranges = new_ranges
        return output

    def resize(
        self,
        width: Union[int, List[int], np.ndarray],
        fix: Literal["start", "end", "center"] = "start",
        ignore_strand: bool = False,
        in_place: bool = False,
    ) -> "RangedSummarizedExperiment":
        """Resize ranges to the specified ``width`` where either the ``start``, ``end``, or ``center`` is used as an
        anchor.

        Args:
            width:
                Width to resize, cannot be negative!

            fix:
                Fix positions by "start", "end", or "center".
                Defaults to "start".

            ignore_strand:
                Whether to ignore strands. Defaults to False.

            in_place:
                Whether to modify the ``GenomicRanges`` object in place.

        Raises:
            ValueError:
                If ``fix`` is neither ``start``, ``center``, or ``end``.

        Returns:
            A new `RangedSummarizedExperiment` object with the resized ranges,
            either as a copy of the original or as a reference to the
            (in-place-modified) original.
        """
        new_ranges = self.row_ranges.resize(
            width=width, fix=fix, ignore_strand=ignore_strand, in_place=False
        )

        output = self._define_output(in_place=in_place)
        output._row_ranges = new_ranges
        return output

    def shift(
        self, shift: Union[int, List[int], np.ndarray] = 0, in_place: bool = False
    ) -> "RangedSummarizedExperiment":
        """Shift all intervals.

        ``shift`` may be be negative.

        Args:
            shift:
                Shift interval. If shift is 0, the current
                object is returned. Defaults to 0.

            in_place:
                Whether to modify the ``GenomicRanges`` object in place.

        Returns:
            A new `RangedSummarizedExperiment` object with the shifted ranges,
            either as a copy of the original or as a reference to the
            (in-place-modified) original.
        """
        new_ranges = self.row_ranges.shift(shift=shift, in_place=False)

        output = self._define_output(in_place=in_place)
        output._row_ranges = new_ranges
        return output

    def promoters(
        self, upstream: int = 2000, downstream: int = 200, in_place: bool = False
    ) -> "RangedSummarizedExperiment":
        """Extend intervals to promoter regions.

        Args:
            upstream:
                Number of positions to extend in the 5' direction.
                Defaults to 2000.

            downstream:
                Number of positions to extend in the 3' direction.
                Defaults to 200.

            in_place:
                Whether to modify the ``GenomicRanges`` object in place.

        Returns:
            A new `RangedSummarizedExperiment` object with the extended ranges for
            promoter regions, either as a copy of the original or as a reference to the
            (in-place-modified) original.
        """
        new_ranges = self.row_ranges.promoters(
            upstream=upstream, downstream=downstream, in_place=False
        )

        output = self._define_output(in_place=in_place)
        output._row_ranges = new_ranges
        return output

    def restrict(
        self,
        start: Optional[Union[int, List[int], np.ndarray]] = None,
        end: Optional[Union[int, List[int], np.ndarray]] = None,
        keep_all_ranges: bool = False,
        in_place: bool = False,
    ) -> "RangedSummarizedExperiment":
        """Restrict ranges to a given start and end positions.

        Args:
            start:
                Start position. Defaults to None.

            end:
                End position. Defaults to None.

            keep_all_ranges:
                Whether to keep intervals that do not overlap with start and end.
                Defaults to False.

            in_place:
                Whether to modify the ``GenomicRanges`` object in place.

        Returns:
            A new `RangedSummarizedExperiment` object with restricted intervals,
            either as a copy of the original or as a reference to the
            (in-place-modified) original.
        """
        new_ranges = self.row_ranges.restrict(
            start=start, end=end, keep_all_ranges=keep_all_ranges, in_place=False
        )

        output = self._define_output(in_place=in_place)
        output._row_ranges = new_ranges
        return output

    def narrow(
        self,
        start: Optional[Union[int, List[int], np.ndarray]] = None,
        width: Optional[Union[int, List[int], np.ndarray]] = None,
        end: Optional[Union[int, List[int], np.ndarray]] = None,
        in_place: bool = False,
    ) -> "RangedSummarizedExperiment":
        """Narrow genomic positions by provided ``start``, ``width`` and ``end`` parameters.

        Important: these parameters are relative shift in positions for each range.

        Args:
            start:
                Relative start position. Defaults to None.

            width:
                Relative end position. Defaults to None.

            end:
                Relative width of the interval. Defaults to None.

            in_place:
                Whether to modify the ``GenomicRanges`` object in place.

        Raises:
            When parameters were set incorrectly or row_ranges is empty

        Returns:
            A new `RangedSummarizedExperiment` object with narrow positions,
            either as a copy of the original or as a reference to the
            (in-place-modified) original.
        """
        new_ranges = self.row_ranges.narrow(
            start=start, width=width, end=end, in_place=False
        )

        output = self._define_output(in_place=in_place)
        output._row_ranges = new_ranges
        return output

    def find_overlaps(
        self,
        query: GRangesOrRangeSE,
        query_type: str = "any",
        select: Literal["all", "first", "last", "arbitrary"] = "all",
        max_gap: int = -1,
        min_overlap: int = 1,
        ignore_strand: bool = False,
    ) -> List[List[int]]:
        """Find overlaps between subject (self) and query ranges.

        Args:
            query:
                Query intervals to find nearest positions.

                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.

            query_type:
                Overlap query type, must be one of

                - "any": Any overlap is good
                - "start": Overlap at the beginning of the intervals
                - "end": Must overlap at the end of the intervals
                - "within": Fully contain the query interval

                Defaults to "any".

            select:
                Determine what hit to choose when
                there are multiple hits for an interval in ``subject``.

            max_gap:
                Maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).

            min_overlap:
                Minimum overlap with query. Defaults to 1.

            ignore_strand:
                Whether to ignore strands. Defaults to False.

        Raises:
            TypeError: If query is not a `RangedSummarizedExperiment` or `GenomicRanges`.

        Returns:
            A list with the same length as ``query``,
            containing hits to overlapping indices.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        return self.row_ranges.find_overlaps(
            query=qranges,
            query_type=query_type,
            select=select,
            max_gap=max_gap,
            min_overlap=min_overlap,
            ignore_strand=ignore_strand,
        )

    def subset_by_overlaps(
        self,
        query: GRangesOrRangeSE,
        query_type: str = "any",
        max_gap: int = -1,
        min_overlap: int = 1,
        ignore_strand: bool = False,
    ) -> "RangedSummarizedExperiment":
        """Subset a `RangedSummarizedExperiment` by feature overlaps.

        Args:
            query:
                Query `GenomicRanges`.

                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.

            query_type:
                Overlap query type, must be one of

                - "any": Any overlap is good
                - "start": Overlap at the beginning of the intervals
                - "end": Must overlap at the end of the intervals
                - "within": Fully contain the query interval

                Defaults to "any".

            max_gap:
                Maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).

            min_overlap:
                Minimum overlap with query. Defaults to 1.

            ignore_strand:
                Whether to ignore strands. Defaults to False.

        Raises:
            TypeError:
                If query is not a `RangedSummarizedExperiment` or `GenomicRanges`.

        Returns:
            A new `RangedSummarizedExperiment`
            object. None if there are no indices to slice.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        result = self.row_ranges.find_overlaps(
            query=qranges,
            query_type=query_type,
            max_gap=max_gap,
            min_overlap=min_overlap,
            ignore_strand=ignore_strand,
        )

        import itertools

        indices = list(itertools.chain.from_iterable(result))

        return self[indices,]

    def order(self, decreasing: bool = False) -> np.ndarray:
        """Get the order of indices to sort.

        Args:
            decreasing:
                Whether to sort in descending order. Defaults to False.

        Returns:
            NumPy vector containing index positions in the sorted order.
        """
        return self.row_ranges.order(decreasing=decreasing)

    def sort(
        self, decreasing: bool = False, in_place: bool = False
    ) -> "RangedSummarizedExperiment":
        """Sort by ranges.

        Args:
            decreasing:
                Whether to sort in descending order. Defaults to False.

            in_place:
                Whether to modify the object in place. Defaults to False.

        Returns:
            A new sorted
            `RangedSummarizedExperiment` object.
        """
        _order = self.row_ranges.order(decreasing=decreasing)

        output = self._define_output(in_place=in_place)
        return output[_order,]
