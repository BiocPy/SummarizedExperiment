from typing import Dict, List, Literal, MutableMapping, Optional, Sequence, Union

import numpy as np
from genomicranges import GenomicRanges, GenomicRangesList, SeqInfo

from .SummarizedExperiment import SummarizedExperiment
from .types import BiocOrPandasFrame, MatrixTypes, SlicerArgTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

GRangesOrGRangesList = Union[GenomicRanges, GenomicRangesList]
GRangesOrRangeSE = Union[GRangesOrGRangesList, "RangedSummarizedExperiment"]


# TODO: technically should be in _type_checks but fails due to circular imports.
def _check_gr_or_rse(x: GRangesOrRangeSE):
    """Check if ``x`` is either a `RangedSummarizedExperiment` or `GenomicRanges`.

    Args:
        x (Union[GenomicRanges, RangedSummarizedExperiment]): Object to check.

    Raises:
        TypeError: Object is not a `RangedSummarizedExperiment` or `GenomicRanges`.
    """
    if not (
        isinstance(x, RangedSummarizedExperiment)
        or isinstance(x, GenomicRanges)
        or isinstance(x, GenomicRangesList)
    ):
        raise TypeError(
            "object is not a `RangedSummarizedExperiment`, `GenomicRanges` or `GenomicRangesList`."
        )


def _access_granges(x: GRangesOrRangeSE) -> GenomicRanges:
    """Access ranges from the object.

    Args:
        x (Union[GenomicRanges, RangedSummarizedExperiment]): Input object.

    Returns:
        GenomicRanges: Genomic ranges object.
    """
    qranges = x
    if isinstance(x, RangedSummarizedExperiment):
        qranges = x.row_ranges

    return qranges


class RangedSummarizedExperiment(SummarizedExperiment):
    """RangedSummarizedExperiment class to represent genomic experiment data, genomic features as
    :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or
    :py:class:`~genomicranges.GenomicRangesList.GenomicRangesList` sample data and any additional experimental metadata.

    The key difference between this and `SummarizedExperiment` is enforcing type for
    feature information (`row_ranges`), must be a `GenomicRanges` object. This allows us to
    provides new methods, to perform genomic range based operations over experimental data.

    Note: If ``row_ranges`` is empty, None or not a
    :py:class:`genomicranges.GenomicRanges.GenomicRanges` object, use a
    :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` instead.

    Attributes:
        assays (MutableMapping[str, MatrixTypes]): Dictionary
            of matrices, with assay names as keys and 2-dimensional matrices represented as
            :py:class:`~numpy.ndarray` or :py:class:`scipy.sparse.spmatrix` matrices.

            Alternatively, you may use any 2-dimensional matrix that contains the property ``shape``
            and implements the slice operation using the ``__getitem__`` dunder method.

            All matrices in ``assays`` must be 2-dimensional and have the same
            shape (number of rows, number of columns).

        row_ranges (GRangesOrGRangesList, optional): Genomic features, must be the same length as
            rows of the matrices in assays.

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
        row_ranges: Optional[GRangesOrGRangesList] = None,
        row_data: Optional[BiocOrPandasFrame] = None,
        col_data: Optional[BiocOrPandasFrame] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize a `RangedSummarizedExperiment` (RSE) object."""
        super().__init__(assays, row_data, col_data, metadata)
        self._validate_row_ranges(row_ranges)
        self._row_ranges = row_ranges

    def _validate_row_ranges(self, row_ranges: GRangesOrGRangesList):
        """Internal method to validate feature information (``row_ranges``).

        Args:
            rows (GRangesOrGRangesList): Genomic features.

        Raises:
            ValueError: When number of rows does not match between `row_ranges` &
                `assays`.
            TypeError: When `row_ranges` is not a `GenomicRanges` or `GenomicRangesList`.
        """
        if not (
            isinstance(row_ranges, GenomicRanges)
            or isinstance(row_ranges, GenomicRangesList)
        ):
            raise TypeError(
                "`row_ranges` must be a `GenomicRanges` or `GenomicRangesList`"
                f" , provided {type(row_ranges)}."
            )

        if len(row_ranges) != self._shape[0]:
            raise ValueError(
                "Number of features and number of rows in assays do not match."
            )

    @property
    def row_ranges(self) -> GRangesOrGRangesList:
        """Get features.

        Returns:
            GRangesOrGRangesList: Genomic features.
        """
        return self._row_ranges

    @row_ranges.setter
    def row_ranges(self, ranges: GRangesOrGRangesList) -> None:
        """Set features.

        Args:
            ranges (GRangesOrGRangesList): Features to update.
        """
        self._validate_row_ranges(ranges)
        self._row_ranges = ranges

    @property
    def end(self) -> List[int]:
        """Get genomic end positions for each feature or row in experimental data.

        Returns:
            List[int]: List of end positions.
        """
        return self.row_ranges.end

    @property
    def start(self) -> List[int]:
        """Get genomic start positions for each feature or row in experimental data.

        Returns:
            List[int]: List of start positions.
        """
        return self.row_ranges.start

    @property
    def seqnames(self) -> List[str]:
        """Get sequence or chromosome names.

        Returns:
            List[str]: List of all chromosome names.
        """
        return self.row_ranges.seqnames

    @property
    def strand(self) -> List[str]:
        """Get strand information.

        Returns:
            List[str]: List of strand information across all features.
        """
        return self.row_ranges.strand

    @property
    def width(self) -> List[int]:
        """Get widths of ``row_ranges``.

        Returns:
            List[int]: List of widths for each interval.
        """
        return self.row_ranges.width

    @property
    def seq_info(self) -> Optional[SeqInfo]:
        """Get sequence information object (if available).

        Returns:
            (SeqInfo, optional): Sequence information.
        """
        return self.row_ranges.seq_info

    def __getitem__(
        self,
        args: SlicerArgTypes,
    ) -> "RangedSummarizedExperiment":
        """Subset a `RangedSummarizedExperiment`.

        Args:
            args (SlicerArgTypes): Indices or names to slice. Tuple contains
                slices along dimensions (rows, cols).

                Each element in the tuple, might be either a integer vector (integer positions),
                boolean vector or :py:class:`~slice` object. Defaults to None.

        Raises:
            ValueError: Too many or few slices.

        Returns:
            RangedSummarizedExperiment: Sliced `RangedSummarizedExperiment` object.
        """
        sliced_objs = self._slice(args)

        new_row_ranges = None
        if sliced_objs.row_indices is not None and self.row_ranges is not None:
            new_row_ranges = self.row_ranges[sliced_objs.row_indices, :]

        return RangedSummarizedExperiment(
            assays=sliced_objs.assays,
            row_ranges=new_row_ranges,
            row_data=sliced_objs.row_data,
            col_data=sliced_objs.col_data,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        pattern = (
            f"Class RangedSummarizedExperiment with {self.shape[0]} features and {self.shape[1]} "
            "samples \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  row_data: {self.row_data.columns if self.row_data is not None else None} \n"
            f"  col_data: {self.col_data.columns if self.col_data is not None else None}"
        )
        return pattern

    def coverage(
        self, shift: int = 0, width: Optional[int] = None, weight: int = 1
    ) -> Dict[str, np.ndarray]:
        """Calculate coverage for each chromosome.

        Args:
            shift (int, optional): Shifts all genomic positions by specified number
                of positions. Defaults to 0.
            width (int, optional): Restrict the width of all chromosomes.
                Defaults to None.
            weight (int, optional): Weight to use. Defaults to 1.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing chromosome names
            as keys and the coverage vector as values.
        """
        return self.row_ranges.coverage(shift=shift, width=width, weight=weight)

    def nearest(
        self,
        query: GRangesOrRangeSE,
        ignore_strand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions, both upstream and downstream that overlap with each range in ``query``.

        Args:
            query (GRangesOrRangeSE): Query intervals to find nearest positions.
                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.

            ignore_strand (bool, optional): Whether to ignore strand during looksups.
                Defaults to False.

        Raises:
            TypeError: If ``query`` is not a ``RangedSummarizedExperiment``
                or ``GenomicRanges``.

        Returns:
            (Sequence[Optional[int]], optional): List of possible `hit` indices
            for each interval in `query`. If there are no hits, returns None.
        """

        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.row_ranges.nearest(query=qranges, ignore_strand=ignore_strand)
        return res.column("hits")

    def precede(
        self,
        query: GRangesOrRangeSE,
        ignore_strand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions, only downstream that overlap with each range in ``query``.

        Args:
            query (GRangesOrRangeSE): Query intervals to find nearest positions.
                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.
            ignore_strand (bool, optional): Whether to ignore strand. Defaults to False.

        Raises:
            TypeError: If ``query`` is not a ``RangedSummarizedExperiment`` or
            ``GenomicRanges``.

        Returns:
            (Sequence[Optional[int]], optional): List of possible hit indices
            for each interval in ``query``. If there are no hits, returns None.
        """

        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.row_ranges.precede(query=qranges, ignore_strand=ignore_strand)
        return res.column("hits")

    def follow(
        self,
        query: GRangesOrRangeSE,
        ignore_strand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions, only upstream that overlap with the each range in ``query``.

        Args:
            query (GRangesOrRangeSE): Query intervals to find nearest positions.
                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.
            ignore_strand (bool, optional): Whether to ignore strand. Defaults to False.

        Raises:
            TypeError: If ``query`` is not a ``RangedSummarizedExperiment`` or
            ``GenomicRanges``.

        Returns:
            (Sequence[Optional[int]], optional): List of possible hit indices
            for each interval in ``query``. If there are no hits, returns None.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.row_ranges.follow(query=qranges, ignore_strand=ignore_strand)
        return res.column("hits")

    def distance_to_nearest(
        self,
        query: GRangesOrRangeSE,
        ignore_strand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions only downstream that overlap with the each genomics interval in ``query``.

        Technically same as
        :py:meth:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment.nearest`
        since we also return `distance` to the nearest match.

        Args:
            query (GRangesOrRangeSE): Query intervals to find nearest positions.
                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.
            ignore_strand (bool, optional): Whether to ignore strand. Defaults to False.

        Raises:
            TypeError: If ``query`` is not a ``RangedSummarizedExperiment`` or ``GenomicRanges``.

        Returns:
            (Sequence[Optional[int]], optional): List of possible hit indices
            for each interval in ``query``. If there are no hits, returns None.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.row_ranges.distance_to_nearest(
            query=qranges, ignore_strand=ignore_strand
        )
        return res.column("distance")

    def flank(
        self,
        width: int,
        start: bool = True,
        both: bool = False,
        ignore_strand: bool = False,
    ) -> "RangedSummarizedExperiment":
        """Generates flanking ranges for each range in
        :py:attr:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment.row_ranges`.

        Refer to either :py:meth:`~genomicranges.GenomicRanges.GenomicRanges.flank` or the
        Bioconductor documentation for what it this method does.

        Args:
            width (int): Width to flank by.
            start (bool, optional): Whether to only flank starts positions. Defaults to True.
            both (bool, optional): Whether to flank both starts and ends positions.
                Defaults to False.
            ignore_strand (bool, optional): Whether to ignore strand. Defaults to False.

        Returns:
            RangedSummarizedExperiment: A new `RangedSummarizedExperiment` object
            with the flanked ranges.
        """
        new_ranges = self.row_ranges.flank(
            width=width, start=start, both=both, ignore_strand=ignore_strand
        )

        return RangedSummarizedExperiment(
            assays=self.assays,
            row_ranges=new_ranges,
            col_data=self.col_data,
            metadata=self.metadata,
        )

    def resize(
        self,
        width: int,
        fix: Literal["start", "end", "center"] = "start",
        ignore_strand: bool = False,
    ) -> "RangedSummarizedExperiment":
        """Resize :py:attr:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment.row_ranges` to
        the specified ``width`` where either the ``start``, ``end``, or ``center`` is used as an anchor.

        Args:
            width (int): Width to resize by.
            fix (Literal["start", "end", "center"], optional): Fix positions by "start", "end"
                or "center". Defaults to "start".
            ignore_strand (bool, optional): Whether to ignore strand. Defaults to False.

        Raises:
            ValueError: If ``fix`` is neither ``start``, ``center``, or ``end``.

        Returns:
            RangedSummarizedExperiment: A new `RangedSummarizedExperiment` object
            with the resized ranges.
        """
        new_ranges = self.row_ranges.resize(
            width=width, fix=fix, ignore_strand=ignore_strand
        )

        return RangedSummarizedExperiment(
            assays=self.assays,
            row_ranges=new_ranges,
            col_data=self.col_data,
            metadata=self.metadata,
        )

    def shift(self, shift: int = 0) -> "RangedSummarizedExperiment":
        """Shift all ranges in
        :py:attr:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment.row_ranges` by the
        specified ``shift`` parameter.

        ``shift`` may be be negative.

        Args:
            shift (int, optional): Shift interval. Defaults to 0.

        Returns:
            RangedSummarizedExperiment: A new `RangedSummarizedExperiment`
            object with the shifted ranges.
        """
        new_ranges = self.row_ranges.shift(shift=shift)

        return RangedSummarizedExperiment(
            assays=self.assays,
            row_ranges=new_ranges,
            col_data=self.col_data,
            metadata=self.metadata,
        )

    def promoters(
        self, upstream: int = 2000, downstream: int = 200
    ) -> "RangedSummarizedExperiment":
        """Extend :py:attr:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment.row_ranges` to
        promoter regions.

        Args:
            upstream (int, optional): Number of positions to extend in the 5' direction.
                Defaults to 2000.
            downstream (int, optional): Number of positions to extend in the 3' direction.
                Defaults to 200.

        Returns:
            RangedSummarizedExperiment: A new `RangedSummarizedExperiment`
            object with the extended ranges for promoter regions.
        """
        new_ranges = self.row_ranges.promoters(upstream=upstream, downstream=downstream)

        return RangedSummarizedExperiment(
            assays=self.assays,
            row_ranges=new_ranges,
            col_data=self.col_data,
            metadata=self.metadata,
        )

    def restrict(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        keep_all_ranges: bool = False,
    ) -> "RangedSummarizedExperiment":
        """Restrict :py:attr:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment.row_ranges` to
        a given start and end positions.

        Args:
            start (int, optional): Start position. Defaults to None.
            end (int, optional): End position. Defaults to None.
            keep_all_ranges (bool, optional): Whether to keep intervals that do not
                overlap with start and end. Defaults to False.

        Returns:
            RangedSummarizedExperiment: A new `RangedSummarizedExperiment`
            object with restricted intervals.
        """
        new_ranges = self.row_ranges.restrict(
            start=start, end=end, keep_all_ranges=keep_all_ranges
        )

        return RangedSummarizedExperiment(
            assays=self.assays,
            row_ranges=new_ranges,
            col_data=self.col_data,
            metadata=self.metadata,
        )

    def narrow(
        self,
        start: Optional[int] = None,
        width: Optional[int] = None,
        end: Optional[int] = None,
    ) -> "RangedSummarizedExperiment":
        """Narrow :py:attr:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment.row_ranges`.

        Important: these parameters are relative shift in positions for each range.

        Args:
            start (int, optional): Relative start position. Defaults to None.
            width (int, optional): Relative end position. Defaults to None.
            end (int, optional): Relative width of the interval. Defaults to None.

        Raises:
            ValueError: When parameters were set incorrectly or row_ranges is empty

        Returns:
            RangedSummarizedExperiment:  A new `RangedSummarizedExperiment`
            object with narrow positions.
        """
        new_ranges = self.row_ranges.narrow(start=start, width=width, end=end)

        return RangedSummarizedExperiment(
            assays=self.assays,
            row_ranges=new_ranges,
            col_data=self.col_data,
            metadata=self.metadata,
        )

    def find_overlaps(
        self,
        query: GRangesOrRangeSE,
        query_type: str = "any",
        max_gap: int = -1,
        min_overlap: int = 1,
        ignore_strand: bool = False,
    ) -> Optional["RangedSummarizedExperiment"]:
        """Find overlaps between subject (self) and query ranges.

        Args:
            query (GRangesOrRangeSE): Query intervals to find overlapping positions.
                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.
            query_type (str, optional): Overlap query type, must be one of

                - "any": any overlap is good
                - "start": overlap at the beginning of the intervals
                - "end": must overlap at the end of the intervals
                - "within": fully contain the query interval

                Defaults to "any".
            max_gap (int, optional): Maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).
            min_overlap (int, optional): Minimum overlap with query. Defaults to 1.
            ignore_strand (bool, optional): Whether to ignore strand.. Defaults to False.

        Raises:
            TypeError: If query is not a `RangedSummarizedExperiment` or `GenomicRanges`.

        Returns:
            ("RangedSummarizedExperiment", optional): A `RangedSummarizedExperiment` object
            with the same length as query, containing hits to overlapping indices.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        return self.row_ranges.find_overlaps(
            query=qranges,
            query_type=query_type,
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
    ) -> Optional["RangedSummarizedExperiment"]:
        """Subset a `RangedSummarizedExperiment` by feature overlaps.

        Args:
            query (GRangesOrRangeSE): Query intervals to subset by overlapping positions.
                ``query`` may be a :py:class:`~genomicranges.GenomicRanges.GenomicRanges` or a
                :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment`
                object.
            query_type (str, optional): Overlap query type, must be one of

                - "any": any overlap is good
                - "start": overlap at the beginning of the intervals
                - "end": must overlap at the end of the intervals
                - "within": fully contain the query interval

                Defaults to "any".
            max_gap (int, optional): Maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).
            min_overlap (int, optional): Minimum overlap with query. Defaults to 1.
            ignore_strand (bool, optional): Whether to ignore strand.. Defaults to False.

        Raises:
            TypeError: If query is not a `RangedSummarizedExperiment` or `GenomicRanges`.

        Returns:
            Optional["RangedSummarizedExperiment"]: A new `RangedSummarizedExperiment`
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

        if result is None:
            return None

        hits = result.column("hits")
        hit_counts = [len(ht) for ht in hits]
        indices = [idx for idx in range(len(hit_counts)) if hit_counts[idx] > 0]

        if indices is None:
            return None

        return self[indices, :]

    def order(self, decreasing=False) -> List[int]:
        """Get the order of indices to sort.

        Args:
            decreasing (bool, optional): Whether to sort in descending order. Defaults to False.

        Returns:
            List[int]: Order of indices.
        """
        return self.row_ranges.order(decreasing=decreasing)

    def sort(
        self, decreasing: bool = False, ignore_strand: bool = False
    ) -> "RangedSummarizedExperiment":
        """Sort `RangedSummarizedExperiment` by
        :py:attr:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment.row_ranges`.

        Args:
            decreasing (bool, optional): Whether to sort in decreasing order. Defaults to False.
            ignore_strand (bool, optional): Whether to ignore strand. Defaults to False.

        Returns:
            "RangedSummarizedExperiment": A new sorted
            `RangedSummarizedExperiment` object.
        """
        order = self.row_ranges._generic_order(ignore_strand=ignore_strand)

        if decreasing:
            order = order[::-1]

        new_order = order.to_list()
        return self[new_order, :]
