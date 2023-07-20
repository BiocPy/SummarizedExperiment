from typing import Union, Tuple, Sequence, Optional, MutableMapping
import pandas as pd
import numpy as np
from scipy import sparse as sp

from biocframe import BiocFrame
from genomicranges import GenomicRanges, SeqInfo

from .BaseSE import BaseSE

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class RangeSummarizedExperiment(BaseSE):
    """RangeSummarizedExperiment class to represent genomic experiment data,
    genomic features, sample data and any other metadata.
    """

    def __init__(
        self,
        assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
        rowRanges: Optional[GenomicRanges] = None,
        colData: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        metadata: Optional[MutableMapping] = None,
    ) -> None:
        """Initialize a new `RangeSummarizedExperiment`.

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): dictionary of matrices,
                with assay names as keys and matrices represented as dense (numpy) or sparse (scipy) matrices.
                All matrices across assays must have the same dimensions (number of rows, number of columns).
            rowRanges (Optional[GenomicRanges], optional): features, must be the same length as rows of the matrices in assays. Defaults to None.
            colData (Optional[Union[pd.DataFrame, BiocFrame]], optional): sample data, must be the same length as rows of the matrices in assays. Defaults to None.
            metadata (Optional[MutableMapping], optional): experiment metadata describing the methods. Defaults to None.
        """
        super().__init__(assays, rowRanges, colData, metadata)

    def _validate_types(self):
        """internal method to validate types
        """
        if self._rows is not None:
            if not (isinstance(self._rows, GenomicRanges)):
                raise TypeError("rowRanges must be a `GenomicRanges` object")

        if self._cols is not None:
            if not (
                isinstance(self._cols, pd.DataFrame)
                or isinstance(self._cols, BiocFrame)
            ):
                raise TypeError(
                    "colData must be either a pandas DataFrame or a `BiocFrame` object"
                )

    @property
    def rowRanges(self) -> Optional[GenomicRanges]:
        """Get features.

        Returns:
            Optional[GenomicRanges]: returns features.
        """
        return self._rows

    @rowRanges.setter
    def rowRanges(self, rows: Optional[GenomicRanges]) -> None:
        """Set features.

        Args:
            rows (Optional[GenomicRanges]): features to update
        """
        self._rows = rows
        self._validate()

    @property
    def end(self) -> Sequence[int]:
        """Get end positions from row ranges.

        Returns:
            Sequence[int]: end locations
        """
        return self.rowRanges.end

    @property
    def start(self) -> Sequence[int]:
        """Get start positions from row ranges

        Returns:
            Sequence[int]: start positions
        """
        return self.rowRanges.start

    @property
    def seqnames(self) -> Sequence[str]:
        """Get sequence or chromosome names

        Returns:
            Sequence[str]: list of all chromosome names
        """
        return self.rowRanges.seqnames

    @property
    def strand(self) -> Optional[Sequence[str]]:
        """Get strand information (if available)

        Returns:
            Optional[Sequence[str]]: strand across all positions or None
        """
        return self.rowRanges.strand

    @property
    def width(self) -> Sequence[int]:
        """Get widths of each interval

        Returns:
            Sequence[int]: width of each interval
        """
        return self.rowRanges.width

    @property
    def seqInfo(self) -> Optional[SeqInfo]:
        """Get the sequence information object (if available)

        Returns:
            SeqInfo: Sequence information
        """
        return self.rowRanges.seqInfo

    def __getitem__(
        self,
        args: Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]],
    ) -> "RangeSummarizedExperiment":
        """Subset a `RangeSummarizedExperiment`.

        Args:
            args (Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]]): indices to slice. tuple can
                contains slices along dimensions

        Raises:
            ValueError: Too many slices

        Returns:
            RangeSummarizedExperiment: new sliced `RangeSummarizedExperiment` object
        """
        new_rows, new_cols, new_assays = self._slice(args)
        return RangeSummarizedExperiment(
            assays=new_assays,
            rowRanges=new_rows,
            colData=new_cols,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        pattern = """
        Class RangeSummarizedExperiment with {} features and {} samples
          assays: {}
          features: {}
          sample data: {}
        """
        return pattern.format(
            self.shape[0],
            self.shape[1],
            list(self._assays.keys()),
            self._rows.columns if self._rows is not None else None,
            self._cols.columns if self._cols is not None else None,
        )

    def coverage(
        self, shift: int = 0, width: Optional[int] = None, weight: int = 1
    ) -> MutableMapping[str, np.ndarray]:
        """Calculate coverage for each chromosome.

        Args:
            shift (int, optional): shift all genomic positions. Defaults to 0.
            width (Optional[int], optional): restrict the width of all chromosomes. Defaults to None.
            weight (int, optional): weight to use. Defaults to 1.

        Raises:
            ValueError: rowRanges is empty

        Returns:
            MutableMapping[str, np.ndarray]: coverage vector for each chromosome.
        """

        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

        return self.rowRanges.coverage(shift=shift, width=width, weight=weight)

    def nearest(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions both upstream and downstream that overlap with the 
            each genomics interval in `query`. Adds a new column to query called `hits`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): input to find nearest positions.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Raises:
            TypeError: if query is not of type `RangeSummarizedExperiment` or `GenomicRanges`
            ValueError: if `rowRanges` is empty
            
        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices for each interval in `query`
        """

        if not (
            isinstance(query, RangeSummarizedExperiment)
            or isinstance(query, GenomicRanges)
        ):
            raise TypeError(
                "query is not a `RangeSummarizedExperiment` or `GenomicRanges`"
            )

        qranges = query
        if isinstance(query, RangeSummarizedExperiment):
            if self.rowRanges is None:
                raise ValueError("rowRanges is None")

            qranges = query.rowRanges

        res = self.rowRanges.nearest(query=qranges, ignoreStrand=ignoreStrand)
        return res.column("hits")

    def precede(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions only downstream that overlap with the 
            each genomics interval in `query`. Adds a new column to query called `hits`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): input to find nearest positions.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Raises:
            TypeError: if query is not of type `RangeSummarizedExperiment` or `GenomicRanges`
            ValueError: if rowRanges is empty
        
        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices for each interval in `query`
        """

        if not (
            isinstance(query, RangeSummarizedExperiment)
            or isinstance(query, GenomicRanges)
        ):
            raise TypeError(
                "query is not a `RangeSummarizedExperiment` or `GenomicRanges`"
            )

        qranges = query
        if isinstance(query, RangeSummarizedExperiment):
            if self.rowRanges is None:
                raise ValueError("rowRanges is None")

            qranges = query.rowRanges

        res = self.rowRanges.precede(query=qranges, ignoreStrand=ignoreStrand)
        return res.column("hits")

    def follow(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions only upstream that overlap with the 
            each genomics interval in `query`. Adds a new column to query called `hits`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): input to find nearest positions.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Raises:
            TypeError: if query is not of type `RangeSummarizedExperiment` or `GenomicRanges`
            ValueError: if rowRanges is empty

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices for each interval in `query`
        """
        if not (
            isinstance(query, RangeSummarizedExperiment)
            or isinstance(query, GenomicRanges)
        ):
            raise TypeError(
                "query is not a `RangeSummarizedExperiment` or `GenomicRanges`"
            )

        qranges = query
        if isinstance(query, RangeSummarizedExperiment):
            if self.rowRanges is None:
                raise ValueError("rowRanges is None")

            qranges = query.rowRanges

        res = self.rowRanges.follow(query=qranges, ignoreStrand=ignoreStrand)
        return res.column("hits")

    def distanceToNearest(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Search nearest positions only downstream that overlap with the 
            each genomics interval in `query`. Adds a new column to query called `hits`.
            Technically same as nearest since we also return `distances`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): input to find nearest positions.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Raises:
            TypeError: if query is not of type `RangeSummarizedExperiment` or `GenomicRanges`
            ValueError: if rowRanges is empty

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices for each interval in `query`
        """
        if not (
            isinstance(query, RangeSummarizedExperiment)
            or isinstance(query, GenomicRanges)
        ):
            raise TypeError(
                "query is not a `RangeSummarizedExperiment` or `GenomicRanges`"
            )

        qranges = query
        if isinstance(query, RangeSummarizedExperiment):
            if self.rowRanges is None:
                raise ValueError("rowRanges is None")

            qranges = query.rowRanges

        res = self.rowRanges.distanceToNearest(query=qranges, ignoreStrand=ignoreStrand)
        return res.column("distance")

    def flank(
        self,
        width: int,
        start: bool = True,
        both: bool = False,
        ignoreStrand: bool = False,
    ) -> "RangeSummarizedExperiment":
        """Flank row ranges.

        Args:
            width (int): width to flank by
            start (bool, optional): only flank starts?. Defaults to True.
            both (bool, optional): both starts and ends?. Defaults to False.
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            ValueError: rowRanges is empty

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with the flanked intervals
        """
        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

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
        self, width: int, fix: str = "start", ignoreStrand: bool = False,
    ) -> "RangeSummarizedExperiment":
        """Resize row ranges

        Args:
            width (int): width to resize
            fix (str, optional): fix positions by `start` or `end`. Defaults to "start".
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            ValueError: parameter fix is neither `start` nor `end` or rowRanges is empty

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with the resized intervals
        """

        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

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
        """Shift row ranges

        Args:
            shift (int, optional): shift interval. Defaults to 0.

        Raises:
            ValueError: rowRanges is empty

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with the shifted intervals
        """

        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

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
        """Extend row ranges to promoter regions.

        Args:
            upstream (int, optional): number of positions to extend in the 5' direction . Defaults to 2000.
            downstream (int, optional): number of positions to extend in the 3' direction. Defaults to 200.

        Raises:
            ValueError: rowRanges is empty

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with the extended intervals for promoter regions
        """

        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

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
        """Restrict row ranges.

        Args:
            start (Optional[int], optional): start position. Defaults to None.
            end (Optional[int], optional): end position. Defaults to None.
            keepAllRanges (bool, optional): Keep intervals that do not overlap with start and end?. Defaults to False.

        Raises:
            ValueError: rowRanges is empty

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with restricted intervals
        """

        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

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

        Args:
            start (Optional[int], optional): relative start position. Defaults to None.
            width (Optional[int], optional): relative end position. Defaults to None.
            end (Optional[int], optional): relative width of the interval. Defaults to None.

        Raises:
            ValueError: when parameters were set incorrectly or rowRanges is empty

        Returns:
            RangeSummarizedExperiment:  a new `RangeSummarizedExperiment` object with narrow positions
        """

        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

        new_ranges = self.rowRanges.narrow(start=start, width=width, end=end)

        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=new_ranges,
            colData=self.colData,
            metadata=self.metadata,
        )

    def findOverlaps(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        queryType: str = "any",
        maxGap: int = -1,
        minOverlap: int = 1,
        ignoreStrand: bool = False,
    ) -> Optional["RangeSummarizedExperiment"]:
        """Find overlaps between subject (self) and a query `RangeSummarizedExperiment`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): query RSE.
            queryType (str, optional): overlap query type, must be one of 
                    "any": any overlap is good
                    "start": overlap at the beginning of the intervals
                    "end": must overlap at the end of the intervals
                    "within": Fully contain the query interval. 
                Defaults to any.
            maxGap (int, optional): maximum gap allowed in the overlap. Defaults to -1 (no gap allowed).
            minOverlap (int, optional): minimum overlap with query. Defaults to 1. 
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            TypeError: if query is not of type `RangeSummarizedExperiment` or `GenomicRanges`
            ValueError: if rowRanges is empty

        Returns:
            Optional["RangeSummarizedExperiment"]: A `RangeSummarizedExperiment` object with the same length as query, 
                containing hits to overlapping indices.
        """

        if not (
            isinstance(query, RangeSummarizedExperiment)
            or isinstance(query, GenomicRanges)
        ):
            raise TypeError(
                "query is not a `RangeSummarizedExperiment` or `GenomicRanges`"
            )

        qranges = query
        if isinstance(query, RangeSummarizedExperiment):
            if self.rowRanges is None:
                raise ValueError("rowRanges is None")

            qranges = query.rowRanges

        return self.rowRanges.findOverlaps(
            query=qranges,
            queryType=queryType,
            maxGap=maxGap,
            minOverlap=minOverlap,
            ignoreStrand=ignoreStrand,
        )

    def subsetByOverlaps(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        queryType: str = "any",
        maxGap: int = -1,
        minOverlap: int = 1,
        ignoreStrand: bool = False,
    ) -> Optional["RangeSummarizedExperiment"]:
        """Subset a `RangeSummarizedExperiment` by feature overlaps.

        Args:
            query (RangeSummarizedExperiment): query `RangeSummarizedExperiment`.
            queryType (str, optional): overlap query type, must be one of 
                    "any": any overlap is good
                    "start": overlap at the beginning of the intervals
                    "end": must overlap at the end of the intervals
                    "within": Fully contain the query interval. 
                Defaults to any.
            maxGap (int, optional): maximum gap allowed in the overlap. Defaults to -1 (no gap allowed).
            minOverlap (int, optional): minimum overlap with query. Defaults to 1. 
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            TypeError: if query is not of type `RangeSummarizedExperiment` or `GenomicRanges`
            ValueError: if rowRanges is empty

        Returns:
            Optional["RangeSummarizedExperiment"]: new `RangeSummarizedExperiment` object
        """

        if not (
            isinstance(query, RangeSummarizedExperiment)
            or isinstance(query, GenomicRanges)
        ):
            raise TypeError(
                "query is not a `RangeSummarizedExperiment` or `GenomicRanges`"
            )

        qranges = query
        if isinstance(query, RangeSummarizedExperiment):
            if self.rowRanges is None:
                raise ValueError("rowRanges is None")

            qranges = query.rowRanges

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
        """Get the order of indices for sorting.

        Args:
            decreasing (bool, optional): descending order?. Defaults to False.

        Raises:
            ValueError: rowRanges is empty

        Returns:
            Sequence[int]: order of indices.
        """

        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

        return self.rowRanges.order(decreasing=decreasing)

    def sort(
        self, decreasing: bool = False, ignoreStrand: bool = False
    ) -> "RangeSummarizedExperiment":
        """Sort the `RangeSummarizedExperiment` by row ranges.

        Args:
            decreasing (bool, optional): decreasing order?. Defaults to False.
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            ValueError: rowRanges is empty

        Returns:
            "RangeSummarizedExperiment": a new sorted `RangeSummarizedExperiment` object.
        """

        if self.rowRanges is None:
            raise ValueError("rowRanges is None")

        order = self.rowRanges._generic_order(ignoreStrand=ignoreStrand)

        if decreasing:
            order = order[::-1]

        new_order = order.to_list()
        return self[new_order, :]
