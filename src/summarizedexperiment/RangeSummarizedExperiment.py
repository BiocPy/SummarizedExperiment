"""The RangedSummarizedExperiment class."""

from typing import Any, Dict, MutableMapping, Optional, Sequence, Tuple, Union

from anndata import AnnData  # type: ignore
from biocframe import BiocFrame  # type: ignore
from genomicranges import GenomicRanges, SeqInfo  # type: ignore
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import spmatrix  # type: ignore

from .BaseSE import BaseSE

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class RangeSummarizedExperiment(BaseSE):
    """RangeSummarizedExperiment class.

    Contains experiment, range feature, sample, and meta- data.
    """

    def __init__(
        self,
        assays: MutableMapping[str, Union[NDArray[Any], spmatrix]],
        rowRanges: Optional[GenomicRanges] = None,
        colData: Optional[Union[DataFrame, BiocFrame]] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        """Initialize a new `RangeSummarizedExperiment`.

        Args:
            assays (MutableMapping[str, NDArray[Any] | spmatrix]): `dict` of
                matrices, with assay names as keys and matrices represented as dense
                (numpy) or sparse (scipy) matrices. All matrices across assays must have
                the same dimensions (number of rows, number of columns).
            rowRanges (GenomicRanges, optional): features, must be the same
                length as rows of the matrices in assays. Defaults to `None`.
            colData (DataFrame, BiocFrame, optional): sample data, must be the same
                length as the columns of the matrices in assays. Defaults to `None`.
            metadata (MutableMapping, optional): experiment metadata describing the
                methods. Defaults to `None`.
        """
        self._shape: Tuple[int, int] = (0, 0)
        self._validate_assays(assays)
        self._assays: MutableMapping[
            str, Union[NDArray[Any], spmatrix]
        ] = assays

        rowRanges = (
            GenomicRanges(numberOfRows=self._shape[0])
            if rowRanges is None
            else rowRanges
        )
        self._validate_rows(rowRanges)

        colData = (
            BiocFrame(numberOfRows=self._shape[1])
            if colData is None
            else colData
        )
        self._validate_cols(colData)

        self._rows = rowRanges
        self._cols = colData
        self._metadata = metadata

    def _validate_rows(self, rowData: GenomicRanges) -> None:
        """Validate row ranges.

        Args:
            rowData (GenomicRanges): row ranges to validate

        Raises:
            ValueError: if row ranges are not valid
        """
        if len(rowData.rowNames) != self._shape[0]:
            raise ValueError(
                "Row ranges must be the same length as rows of the matrices in assays: "
                f"{self._shape[0]}"
            )

    @property
    def rowRanges(self) -> GenomicRanges:
        """Get features.

        Returns:
            Optional[GenomicRanges]: returns features.
        """
        return self._rows

    @rowRanges.setter
    def rowRanges(self, rows: GenomicRanges) -> None:
        """Set features.

        Args:
            rows (GenomicRanges): new features.
        """
        self._validate_rows(rows)
        self._rows = rows

    @property
    def rownames(self) -> Sequence[str]:
        """Get row/feature names.

        Returns:
            Sequence[str]: list of row index names
        """
        return self._rows.rownames

    def _slice(
        self, args: Tuple[Union[Sequence[int], slice], ...]
    ) -> Tuple[
        MutableMapping[str, Union[NDArray[Any], spmatrix]],
        Any,
        Union[DataFrame, BiocFrame],
    ]:
        """Internal method to slice `SE` by index.

        Args:
            args (Tuple[Union[Sequence[int], slice], ...]): Indices to slice, `tuple`
                can contain slices along dimensions, max 2 dimensions accepted.

        Raises:
            ValueError: Too many slices

        Returns:
            MutableMapping[str, Union[np.ndarray, spmatrix]]: Sliced assays.
            Any: Sliced rows.
            Union[DataFrame, BiocFrame]: Sliced cols.
        """
        if len(args) == 0:
            raise ValueError("Arguments must contain at least one slice.")

        rowIndices = args[0]
        colIndices = None

        if len(args) > 1:
            colIndices = args[1]
        elif len(args) > 2:
            raise ValueError("Parameter 'args' contains too many slices.")

        new_rows = self._rows[rowIndices, :]  # type: ignore

        new_cols: Union[DataFrame, BiocFrame] = self._cols
        if colIndices is not None:
            if isinstance(self._cols, DataFrame):
                new_cols = self._cols.iloc[colIndices]  # type: ignore
            else:
                new_cols = self._cols[colIndices, :]

        new_assays = self.subsetAssays(
            rowIndices=rowIndices, colIndices=colIndices
        )

        return (new_assays, new_rows, new_cols)

    def __str__(self) -> str:
        """The string representation of the experiment."""
        pattern = (
            "Class: RangedSummarizedExperiment with {} features and {} samples\n"
            "   assays: {}\n"
            "   features: {}\n"
            "   sample data: {}\n"
        )

        return pattern.format(
            self.shape[0],
            self.shape[1],
            list(self._assays.keys()),
            self._rows.columns,
            self._cols.columns,
        )

    def __getitem__(
        self,
        args: Tuple[Union[Sequence[int], slice], ...],
    ) -> "RangeSummarizedExperiment":
        """Subset a `RangeSummarizedExperiment`.

        Args:
            args (Tuple[Union[Sequence[int], slice], ...]): Indices to slice, `tuple`
                can contain slices along dimensions, max 2 dimensions accepted.

        Raises:
            ValueError: Too many slices

        Returns:
            RangeSummarizedExperiment: new sliced `RangeSummarizedExperiment` object
        """
        return RangeSummarizedExperiment(*self._slice(args), self.metadata)

    def toAnnData(self) -> AnnData:
        """Transform `SingleCellExperiment` object to `AnnData`.

        Returns:
            AnnData: return an `AnnData` representation of SE.
        """
        layers: Dict[str, Any] = {}
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()  # type: ignore

        return AnnData(
            obs=self.colData
            if isinstance(self.colData, DataFrame)
            else self.colData.toPandas(),
            var=self.rowRanges.toPandas(),  # type: ignore
            uns=self.metadata,
            layers=layers,
        )

    @property
    def end(self) -> Sequence[int]:
        """Get end positions from row ranges.

        Returns:
            Sequence[int]: end locations
        """
        return self.rowRanges.end

    @property
    def start(self) -> Sequence[int]:
        """Get start positions from row ranges.

        Returns:
            Sequence[int]: start positions
        """
        return self.rowRanges.start

    @property
    def seqnames(self) -> Sequence[str]:
        """Get sequence or chromosome names.

        Returns:
            Sequence[str]: list of all chromosome names
        """
        return self.rowRanges.seqnames

    @property
    def strand(self) -> Optional[Sequence[str]]:
        """Get strand information (if available).

        Returns:
            Optional[Sequence[str]]: strand across all positions or None
        """
        return self.rowRanges.strand

    @property
    def width(self) -> Sequence[int]:
        """Get widths of each interval.

        Returns:
            Sequence[int]: width of each interval
        """
        return self.rowRanges.width

    @property
    def seqInfo(self) -> Optional[SeqInfo]:
        """Get the sequence information object (if available).

        Returns:
            SeqInfo: Sequence information
        """
        return self.rowRanges.seqInfo

    def coverage(
        self, shift: int = 0, width: Optional[int] = None, weight: int = 1
    ) -> MutableMapping[str, NDArray[Any]]:
        """Calculate coverage for each chromosome.

        Args:
            shift (int, optional): shift all genomic positions. Defaults to 0.
            width (Optional[int], optional): restrict the width of all chromosomes.
                Defaults to None.
            weight (int, optional): weight to use. Defaults to 1.

        Raises:
            ValueError: rowRanges is empty

        Returns:
            MutableMapping[str, NDArray[Any]]: coverage vector for each chromosome.
        """
        return self.rowRanges.coverage(shift, width, weight)  # type: ignore

    def nearest(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Find the nearest (up & down)stream intervals overlapping query intervals.

        Search the nearest positions both upstream and downstream that overlap with the
        genomic intervals in `query`. Adds a new column to query called `hits`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): Query intervals
                to find overlap for.
            ignoreStrand (bool, optional): ignore strand? Default is False.

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices for each
                interval in `query`
        """
        return self.rowRanges.nearest(
            query=query.rowRanges
            if isinstance(query, RangeSummarizedExperiment)
            else query,
            ignoreStrand=ignoreStrand,
        ).column(  # type: ignore
            "hits"
        )

    def precede(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Find the nearest downstream intervals overlapping query intervals.

        Search the nearest positions downstream that overlap with the genomic intervals
        in `query`. Adds a new column to query called `hits`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): Query intervals
                to find overlap for.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices for each
                interval in `query`
        """
        return self.rowRanges.precede(
            query=query.rowRanges
            if isinstance(query, RangeSummarizedExperiment)
            else query,
            ignoreStrand=ignoreStrand,
        ).column(  # type: ignore
            "hits"
        )

    def follow(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Find the nearest upstream intervals overlapping query intervals.

        Search the nearest positions upstream that overlap with the genomic intervals
        in `query`. Adds a new column to query called `hits`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): Query intervals
                to find overlap for.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices for each
                interval in `query`
        """
        return self.rowRanges.follow(
            query=query.rowRanges
            if isinstance(query, RangeSummarizedExperiment)
            else query,
            ignoreStrand=ignoreStrand,
        ).column(  # type: ignore
            "hits"
        )

    def distanceToNearest(
        self,
        query: Union[GenomicRanges, "RangeSummarizedExperiment"],
        ignoreStrand: bool = False,
    ) -> Optional[Sequence[Optional[int]]]:
        """Same as `precede` but with distances.

        Finds nearest downstream positions that overlap with intervals in `query`.
        Adds a new column to query called `hits`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): Query intervals
                to find overlap for.
            ignoreStrand (bool, optional): ignore strand? Defaults to False.


        Raises:
            ValueError: if `both` and `start` are both True.

        Returns:
            Optional[Sequence[Optional[int]]]: List of possible hit indices for each
                interval in `query`.
        """
        return self.rowRanges.distanceToNearest(
            query=query.rowRanges
            if isinstance(query, RangeSummarizedExperiment)
            else query,
            ignoreStrand=ignoreStrand,
        ).column(  # type: ignore
            "distance"
        )

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
            ValueError: if `both` is True and `start` is False

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with
                the flanked intervals
        """
        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=self.rowRanges.flank(
                width=width, start=start, both=both, ignoreStrand=ignoreStrand
            ),
            colData=self.colData,
            metadata=self.metadata,
        )

    def resize(
        self, width: int, fix: str = "start", ignoreStrand: bool = False
    ) -> "RangeSummarizedExperiment":
        """Resize row ranges.

        Args:
            width (int): width to resize
            fix (str, optional): fix positions by `start` or `end`. Defaults to "start".
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            ValueError: if `fix` is not `start` or `end`

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with the
                resized intervals
        """
        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=self.rowRanges.resize(
                width=width, fix=fix, ignoreStrand=ignoreStrand
            ),
            colData=self.colData,
            metadata=self.metadata,
        )

    def shift(self, shift: int = 0) -> "RangeSummarizedExperiment":
        """Shift row ranges.

        Args:
            shift (int, optional): shift interval. Defaults to 0.

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with the shifted intervals
        """
        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=self.rowRanges.shift(shift=shift),
            colData=self.colData,
            metadata=self.metadata,
        )

    def promoters(
        self, upstream: int = 2000, downstream: int = 200
    ) -> "RangeSummarizedExperiment":
        """Extend row ranges to promoter regions.

        Args:
            upstream (int, optional): number of positions to extend in the 5' direction.
                Defaults to 2000.
            downstream (int, optional): number of positions to extend in the 3'
                direction. Defaults to 200.


        Raises:
            ValueError: `upstream` or `downstream` is negative.

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with the
                extended intervals for promoter regions
        """
        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=self.rowRanges.promoters(
                upstream=upstream, downstream=downstream
            ),
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
            keepAllRanges (bool, optional): Keep intervals that do not overlap with
                start and end? Defaults to False.

        Raises:
            ValueError: `start` and `end` are both None.

        Returns:
            RangeSummarizedExperiment: a new `RangeSummarizedExperiment` object with
                restricted intervals
        """
        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=self.rowRanges.restrict(
                start=start, end=end, keepAllRanges=keepAllRanges
            ),
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
            end (Optional[int], optional): relative width of the interval. Defaults to
                None.

        Raises:
            ValueError: `start` and `end` are both None or `start` and `width` are both
                None.

        Returns:
            RangeSummarizedExperiment:  a new `RangeSummarizedExperiment` object with
                narrow positions
        """
        return RangeSummarizedExperiment(
            assays=self.assays,
            rowRanges=self.rowRanges.narrow(start=start, width=width, end=end),
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
    ) -> Optional[GenomicRanges]:
        """Find overlaps between subject (self) and a query `RangeSummarizedExperiment`.

        Args:
            query (Union[GenomicRanges, "RangeSummarizedExperiment"]): query RSE.
            queryType (str, optional): overlap query type, must be one of
                    "any": any overlap is good
                    "start": overlap at the beginning of the intervals
                    "end": must overlap at the end of the intervals
                    "within": Fully contain the query interval.
                Defaults to "any".
            maxGap (int, optional): maximum gap allowed in the overlap. Defaults to -1 (no gap allowed).
            minOverlap (int, optional): minimum overlap with query. Defaults to 1.
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Raises:
            ValueError: if `queryType` is not one of "any", "start", "end", "within".

        Returns:
            (GenomicRanges, optional): A `GenomicRangesObject` object with the same
            length as query, containing hits to overlapping indices.
        """
        return self.rowRanges.findOverlaps(
            query=query.rowRanges
            if isinstance(query, RangeSummarizedExperiment)
            else query,
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
                Defaults to "any".
            maxGap (int, optional): maximum gap allowed in the overlap. Defaults to -1 (no gap allowed).
            minOverlap (int, optional): minimum overlap with query. Defaults to 1.
            ignoreStrand (bool, optional): ignore strand?. Defaults to False.

        Returns:
            Optional["RangeSummarizedExperiment"]: new `RangeSummarizedExperiment` object
        """
        result = self.rowRanges.findOverlaps(
            query=query.rowRanges
            if isinstance(query, RangeSummarizedExperiment)
            else query,
            queryType=queryType,
            maxGap=maxGap,
            minOverlap=minOverlap,
            ignoreStrand=ignoreStrand,
        )

        if result is not None:
            hit_counts = [len(hit) for hit in result.column("hits")]  # type: ignore
            indices = [
                idx for idx in range(len(hit_counts)) if hit_counts[idx] > 0
            ]
            result = self[indices, :]

        return result

    def order(self, decreasing: bool = False) -> Sequence[int]:
        """Get the order of indices for sorting.

        Args:
            decreasing (bool, optional): descending order?. Defaults to False.

        Returns:
            Sequence[int]: order of indices.
        """
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
        order = self.rowRanges._generic_order(ignoreStrand=ignoreStrand)  # type: ignore

        if decreasing:
            order = order[::-1]

        return self[order.to_list(), :]  # type: ignore
