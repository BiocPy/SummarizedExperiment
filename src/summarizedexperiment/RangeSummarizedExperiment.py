from typing import MutableMapping, Optional, Sequence, Union

import numpy as np
from genomicranges import GenomicRanges, SeqInfo

from .BaseSE import BaseSE
from .types import BiocOrPandasFrame, MatrixTypes, SlicerArgTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

GRangesOrRangeSE = Union[GenomicRanges, "RangeSummarizedExperiment"]


# TODO: technically should be in _type_checks but fails due to circular imports.
def _check_gr_or_rse(x: GRangesOrRangeSE):
    """Check if the object is either a `RangeSummarizedExperiment` or `GenomicRanges`.

    Args:
        x (Union[GenomicRanges, RangeSummarizedExperiment]): object to check.

    Raises:
        TypeError: object is not a `RangeSummarizedExperiment` or `GenomicRanges`.
    """
    if not (isinstance(x, RangeSummarizedExperiment) or isinstance(x, GenomicRanges)):
        raise TypeError(
            "object is not a `RangeSummarizedExperiment` or `GenomicRanges`"
        )


def _access_granges(x: GRangesOrRangeSE) -> GenomicRanges:
    """Access ranges from the object.

    Args:
        x (Union[GenomicRanges, RangeSummarizedExperiment]): input object.

    Returns:
        GenomicRanges: genomic ranges object.
    """
    qranges = x
    if isinstance(x, RangeSummarizedExperiment):
        qranges = x.rowRanges

    return qranges


class RangeSummarizedExperiment(BaseSE):
    """RangeSummarizedExperiment class to represent genomic experiment data,
    genomic features (as GenomicRanges), sample data and any other metadata.
    """

    def __init__(
        self,
        assays: MutableMapping[str, MatrixTypes],
        rowRanges: Optional[GenomicRanges],
        rowData: Optional[BiocOrPandasFrame] = None,
        colData: Optional[BiocOrPandasFrame] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize a Range Summarized Experiment (RSE) object.

        The key difference between this and `SummarizedExperiment` is enforcing
        type for feature information (`rowRanges`), must be a `GenomicRanges` object.
        This allows us to provides new methods, to perform genomic range based
        operations over experimental data.

        Note: If `rowRanges` is empty, None or not a genomic ranges object,
        use a `SummarizedExperiment` instead!

        Args:
            assays (MutableMapping[str, MatrixTypes]): dictionary
                of matrices, with assay names as keys and matrices represented as dense
                (numpy) or sparse (scipy) matrices. All matrices across assays must
                have the same dimensions (number of rows, number of columns).
            rowRanges (GenomicRanges): genomic ranges, must be the same length as
                rows of the matrices in assays.
            rowData (BiocOrPandasFrame, optional): features, must be the
                same length as rows of the matrices in assays. Defaults to None.
            colData (BiocOrPandasFrame, optional): sample data, must be
                the same length as columns of the matrices in assays. Defaults to None.
            metadata (MutableMapping, optional): experiment metadata describing the
                methods. Defaults to None.
        """
        super().__init__(assays, rowData, colData, metadata)
        self._validate_rowRanges(rowRanges)
        self._rowRanges = rowRanges

    def _validate_rowRanges(self, rowsRanges: GenomicRanges):
        """Internal method to validate feature information (`rowRanges`).

        Args:
            rows (GenomicRanges): genomic features (rowRanges).

        Raises:
            ValueError: when number of rows does not match between `rowRanges` &
                `assays`.
            TypeError: when `rowRanges` is not a `GenomicRanges` object.
        """
        if not (isinstance(rowsRanges, GenomicRanges)):
            raise TypeError(
                "rowsRanges must be a `GenomicRanges`"
                f" object, provided {type(rowsRanges)}"
            )

        if rowsRanges.shape[0] != self._shape[0]:
            raise ValueError(
                f"Features and assays do not match. must be {self._shape[0]}"
                f" but provided {rowsRanges.shape[0]}"
            )

    @property
    def rowRanges(self) -> GenomicRanges:
        """Get features.

        Returns:
            GenomicRanges: returns features.
        """
        return self._rowRanges

    @rowRanges.setter
    def rowRanges(self, ranges: GenomicRanges) -> None:
        """Set features.

        Args:
            ranges (GenomicRanges): features to update.
        """
        self._validate_rowRanges(ranges)
        self._rowRanges = ranges

    @property
    def end(self) -> Sequence[int]:
        """Get genomic end positions for each feature or row in
        experimental data.

        Returns:
            Sequence[int]: end positions.
        """
        return self.rowRanges.end

    @property
    def start(self) -> Sequence[int]:
        """Get genomic start positions for each feature or row in
        experimental data.

        Returns:
            Sequence[int]: start positions.
        """
        return self.rowRanges.start

    @property
    def seqnames(self) -> Sequence[str]:
        """Get sequence or chromosome names.

        Returns:
            Sequence[str]: list of all chromosome names.
        """
        return self.rowRanges.seqnames

    @property
    def strand(self) -> Sequence[str]:
        """Get strand information.

        Returns:
            Sequence[str]: strand across all features.
        """
        return self.rowRanges.strand

    @property
    def width(self) -> Sequence[int]:
        """Get widths of each feature.

        Returns:
            Sequence[int]: width of each feature.
        """
        return self.rowRanges.width

    @property
    def seqInfo(self) -> Optional[SeqInfo]:
        """Get the sequence information object (if available).

        Returns:
            SeqInfo: Sequence information.
        """
        return self.rowRanges.seqInfo

    def __getitem__(
        self,
        args: SlicerArgTypes,
    ) -> "RangeSummarizedExperiment":
        """Subset a `RangeSummarizedExperiment`.

        Args:
            args (SlicerArgTypes): indices to slice.
                tuple can contains slices along row and column dimensions.

        Raises:
            ValueError: Too many or few slices.

        Returns:
            RangeSummarizedExperiment: Sliced `RangeSummarizedExperiment` object.
        """
        sliced_objs = self._slice(args)

        new_rowRanges = None
        if sliced_objs.rowIndices is not None and self.rowRanges is not None:
            new_rowRanges = self.rowRanges[sliced_objs.rowIndices, :]

        return RangeSummarizedExperiment(
            assays=sliced_objs.assays,
            rowRanges=new_rowRanges,
            rowData=sliced_objs.rowData,
            colData=sliced_objs.colData,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        pattern = (
            f"Class RangeSummarizedExperiment with {self.shape[0]} features and {self.shape[1]} samples \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  features: {self.rowData.columns if self.rowData is not None else None} \n"
            f"  sample data: {self.colData.columns if self.colData is not None else None}"
        )
        return pattern

    def coverage(
        self, shift: int = 0, width: Optional[int] = None, weight: int = 1
    ) -> MutableMapping[str, np.ndarray]:
        """Calculate coverage for each chromosome.

        Args:
            shift (int, optional): shifts all genomic positions by specified number
                of positions. Defaults to 0.
            width (int, optional): restrict the width of all chromosomes.
                Defaults to None.
            weight (int, optional): weight to use. Defaults to 1.

        Returns:
            MutableMapping[str, np.ndarray]: a dictionary containing chromosome names
            as keys and the coverage vector as values.
        """
        return self.rowRanges.coverage(shift=shift, width=width, weight=weight)

    def nearest(
        self,
        query: GRangesOrRangeSE,
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions, both upstream and downstream that overlap with
        each range in `query`.

        Args:
            query (GRangesOrRangeSE): query intervals
                to find nearest positions.
            ignoreStrand (bool, optional): ignore strand during looksups?
                Defaults to False.

        Raises:
            TypeError: if `query` is not a `RangeSummarizedExperiment`
                or `GenomicRanges`.

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices
            for each interval in `query`. If there are no hits, returns None.
        """

        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.rowRanges.nearest(query=qranges, ignoreStrand=ignoreStrand)
        return res.column("hits")

    def precede(
        self,
        query: GRangesOrRangeSE,
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions, only downstream that overlap with
        each range in `query`.

        Args:
            query (GRangesOrRangeSE): query intervals
                to find nearest positions.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Raises:
            TypeError: if query is not a `RangeSummarizedExperiment` or
            `GenomicRanges`.

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices
            for each interval in `query`. If there are no hits, returns None.
        """

        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.rowRanges.precede(query=qranges, ignoreStrand=ignoreStrand)
        return res.column("hits")

    def follow(
        self,
        query: GRangesOrRangeSE,
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions, only upstream that overlap with the
        each range in `query`.

        Args:
            query (GRangesOrRangeSE): query intervals
                to find nearest positions.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Raises:
            TypeError: if query is not a `RangeSummarizedExperiment` or
            `GenomicRanges`.

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices
            for each interval in `query`. If there are no hits, returns None.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.rowRanges.follow(query=qranges, ignoreStrand=ignoreStrand)
        return res.column("hits")

    def distanceToNearest(
        self,
        query: GRangesOrRangeSE,
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions only downstream that overlap with the
        each genomics interval in `query`.

        Technically same as `nearest` since we also return `distance` to the
        nearest match.

        Args:
            query (GRangesOrRangeSE): query intervals
                to find nearest positions.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Raises:
            TypeError: if query is not a `RangeSummarizedExperiment` or `GenomicRanges`.

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices
            for each interval in `query`. If there are no hits, returns None.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        res = self.rowRanges.distanceToNearest(query=qranges, ignoreStrand=ignoreStrand)
        return res.column("distance")

    def flank(
        self,
        width: int,
        start: bool = True,
        both: bool = False,
        ignoreStrand: bool = False,
    ) -> "RangeSummarizedExperiment":
        """Generates flanking ranges for each range in `rowRanges`.

        Refer to either `GenomicRanges` or the Bioconductor documentation
        for what it this method does.

        Args:
            width (int): width to flank by.
            start (bool, optional): only flank starts?. Defaults to True.
            both (bool, optional): both starts and ends?. Defaults to False.
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object
            with the flanked ranges.
        """
        new_ranges = self.rowRanges.flank(
            width=width, start=start, both=both, ignoreStrand=ignoreStrand
        )

        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=new_ranges,
            colData=self.colData,
            metadata=self.metadata,
        )

    def resize(
        self,
        width: int,
        fix: str = "start",
        ignoreStrand: bool = False,
    ) -> "RangeSummarizedExperiment":
        """Resize `rowRanges` to the specified `width` where either the `start`,
        `end`, or `center` is used as an anchor.

        Args:
            width (int): width to resize by.
            fix (str, optional): fix positions by `start`, `end` or `center`.
                Defaults to "start".
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            ValueError: parameter fix is neither `start`, `cetner`, or `end`.

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object
            with the resized ranges.
        """
        new_ranges = self.rowRanges.resize(
            width=width, fix=fix, ignoreStrand=ignoreStrand
        )

        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=new_ranges,
            colData=self.colData,
            metadata=self.metadata,
        )

    def shift(self, shift: int = 0) -> "RangeSummarizedExperiment":
        """Shift all ranges in `rowRanges` by the specified `shift` parameter.

        shift can be be a negative parameter.

        Args:
            shift (int, optional): shift interval. Defaults to 0.

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment`
            object with the shifted ranges.
        """
        new_ranges = self.rowRanges.shift(shift=shift)

        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=new_ranges,
            colData=self.colData,
            metadata=self.metadata,
        )

    def promoters(
        self, upstream: int = 2000, downstream: int = 200
    ) -> "RangeSummarizedExperiment":
        """Extend `rowRanges` to promoter regions.

        Args:
            upstream (int, optional): number of positions to extend
                in the 5' direction . Defaults to 2000.
            downstream (int, optional): number of positions to extend
                in the 3' direction. Defaults to 200.

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment`
            object with the extended ranges for promoter regions.
        """
        new_ranges = self.rowRanges.promoters(upstream=upstream, downstream=downstream)

        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=new_ranges,
            colData=self.colData,
            metadata=self.metadata,
        )

    def restrict(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        keepAllRanges: bool = False,
    ) -> "RangeSummarizedExperiment":
        """Restrict `rowRanges` to a given start and end positions.

        Args:
            start (int, optional): start position. Defaults to None.
            end (int, optional): end position. Defaults to None.
            keepAllRanges (bool, optional): Keep intervals that do not
                overlap with start and end?. Defaults to False.

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment`
            object with restricted intervals.
        """
        new_ranges = self.rowRanges.restrict(
            start=start, end=end, keepAllRanges=keepAllRanges
        )

        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=new_ranges,
            colData=self.colData,
            metadata=self.metadata,
        )

    def narrow(
        self,
        start: Optional[int] = None,
        width: Optional[int] = None,
        end: Optional[int] = None,
    ) -> "RangeSummarizedExperiment":
        """Narrow row ranges.

        Important: these parameters are relative shift in positions for each range.

        Args:
            start (int, optional): relative start position. Defaults to None.
            width (int, optional): relative end position. Defaults to None.
            end (int, optional): relative width of the interval. Defaults to None.

        Raises:
            ValueError: when parameters were set incorrectly or rowRanges is empty

        Returns:
            RangeSummarizedExperiment:  a new `RangeSummarizedExperiment`
            object with narrow positions.
        """
        new_ranges = self.rowRanges.narrow(start=start, width=width, end=end)

        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=new_ranges,
            colData=self.colData,
            metadata=self.metadata,
        )

    def findOverlaps(
        self,
        query: GRangesOrRangeSE,
        queryType: str = "any",
        maxGap: int = -1,
        minOverlap: int = 1,
        ignoreStrand: bool = False,
    ) -> Optional["RangeSummarizedExperiment"]:
        """Find overlaps between subject (self) and query ranges.

        Args:
            query (GRangesOrRangeSE): query ranges.
            queryType (str, optional): overlap query type, must be one of
                - "any": any overlap is good.
                - "start": overlap at the beginning of the intervals.
                - "end": must overlap at the end of the intervals.
                - "within": fully contain the query interval.
                Defaults to any.
            maxGap (int, optional): maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).
            minOverlap (int, optional): minimum overlap with query. Defaults to 1.
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            TypeError: if query is not a `RangeSummarizedExperiment` or `GenomicRanges`.

        Returns:
            Optional["RangeSummarizedExperiment"]: A `RangeSummarizedExperiment` object
            with the same length as query, containing hits to overlapping indices.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        return self.rowRanges.findOverlaps(
            query=qranges,
            queryType=queryType,
            maxGap=maxGap,
            minOverlap=minOverlap,
            ignoreStrand=ignoreStrand,
        )

    def subsetByOverlaps(
        self,
        query: GRangesOrRangeSE,
        queryType: str = "any",
        maxGap: int = -1,
        minOverlap: int = 1,
        ignoreStrand: bool = False,
    ) -> Optional["RangeSummarizedExperiment"]:
        """Subset a `RangeSummarizedExperiment` by feature overlaps.

        Args:
            query (GRangesOrRangeSE): query ranges.
            queryType (str, optional): overlap query type, must be one of
                - "any": any overlap is good.
                - "start": overlap at the beginning of the intervals.
                - "end": must overlap at the end of the intervals.
                - "within": fully contain the query interval.
                Defaults to any.
            maxGap (int, optional): maximum gap allowed in the overlap.
                Defaults to -1 (no gap allowed).
            minOverlap (int, optional): minimum overlap with query. Defaults to 1.
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            TypeError: if query is not a `RangeSummarizedExperiment` or `GenomicRanges`.

        Returns:
            Optional["RangeSummarizedExperiment"]: new `RangeSummarizedExperiment`
            object. None if there are no indices to slice.
        """
        _check_gr_or_rse(query)

        qranges = _access_granges(query)

        result = self.rowRanges.findOverlaps(
            query=qranges,
            queryType=queryType,
            maxGap=maxGap,
            minOverlap=minOverlap,
            ignoreStrand=ignoreStrand,
        )

        if result is None:
            return None

        hits = result.column("hits")
        hit_counts = [len(ht) for ht in hits]
        indices = [idx for idx in range(len(hit_counts)) if hit_counts[idx] > 0]

        return self[indices, :]

    def order(self, decreasing=False) -> Sequence[int]:
        """Get the order of indices to sort.

        Args:
            decreasing (bool, optional): descending order?. Defaults to False.

        Returns:
            Sequence[int]: order of indices.
        """
        return self.rowRanges.order(decreasing=decreasing)

    def sort(
        self, decreasing: bool = False, ignoreStrand: bool = False
    ) -> "RangeSummarizedExperiment":
        """Sort `RangeSummarizedExperiment` by `rowRanges`.

        Args:
            decreasing (bool, optional): decreasing order?. Defaults to False.
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Returns:
            "RangeSummarizedExperiment": a new sorted
            `RangeSummarizedExperiment` object.
        """
        order = self.rowRanges._generic_order(ignoreStrand=ignoreStrand)

        if decreasing:
            order = order[::-1]

        new_order = order.to_list()
        return self[new_order, :]
