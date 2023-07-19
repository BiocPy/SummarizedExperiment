from random import random

import genomicranges
import numpy as np
import pandas as pd
import pytest
from summarizedexperiment.RangeSummarizedExperiment import RangeSummarizedExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


nrows = 200
ncols = 6
counts = np.random.rand(nrows, ncols)
df_gr = pd.DataFrame(
    {
        "seqnames": [
            "chr1",
            "chr2",
            "chr2",
            "chr2",
            "chr1",
            "chr1",
            "chr3",
            "chr3",
            "chr3",
            "chr3",
        ]
        * 20,
        "starts": range(100, 300),
        "ends": range(110, 310),
        "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * 20,
        "score": range(0, 200),
        "GC": [random() for _ in range(10)] * 20,
    }
)

gr = genomicranges.fromPandas(df_gr)

colData = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_RSE_props():
    tse = RangeSummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, RangeSummarizedExperiment)

    assert tse.assayNames is not None
    assert len(tse.assayNames) == 1

    assert tse.colData is not None
    assert tse.rowRanges is not None

    assert tse.dims == tse.shape

    assert tse.metadata is None

    assert tse.start is not None
    assert tse.end is not None

    assert tse.seqInfo is None
    assert tse.seqnames is not None
    assert tse.strand is not None
    assert tse.width is not None

    assert tse.rownames is None
    assert tse.colnames is not None


def test_RSE_subset():
    tse = RangeSummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, RangeSummarizedExperiment)

    subset_tse = tse[0:10, 2:5]
    assert subset_tse is not None
    assert isinstance(subset_tse, RangeSummarizedExperiment)

    assert len(subset_tse.rowRanges) == 10
    assert len(subset_tse.colData) == 3

    assert subset_tse.assay("counts").shape == (10, 3)


def test_RSE_subsetAssays():
    tse = RangeSummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, RangeSummarizedExperiment)

    subset_asys = tse.subsetAssays(rowIndices=slice(1, 10), colIndices=[0, 1, 2])
    assert subset_asys is not None
    assert isinstance(subset_asys, type(tse.assays))

    assert len(subset_asys.keys()) == 1
    assert subset_asys["counts"].shape == (9, 3)


def test_RSE_coverage():
    tse = RangeSummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, RangeSummarizedExperiment)

    cov = tse.coverage()
    assert cov is not None
    assert len(cov.keys()) == 3


def test_RSE_distance_methods():
    tse = RangeSummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, RangeSummarizedExperiment)

    nearest = tse.nearest(tse)
    assert nearest is not None

    precede = tse.precede(tse)
    assert precede is not None

    follow = tse.follow(tse)
    assert follow is not None

    distNear = tse.distanceToNearest(tse)
    assert distNear is not None


def test_RSE_range_methods():
    tse = RangeSummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, RangeSummarizedExperiment)

    res = tse.flank(width=10)
    assert res is not None

    res = tse.resize(width=15)
    assert res is not None

    with pytest.raises(Exception):
        res = tse.restrict(start=400)

    res = tse.shift(shift=25)
    assert res is not None

    res = tse.promoters()
    assert res is not None

    res = tse.narrow(start=40)
    assert res is not None


def test_RSE_search():
    tse = RangeSummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, RangeSummarizedExperiment)

    res = tse.findOverlaps(tse)
    assert res is not None

    res = tse.subsetByOverlaps(tse)
    assert res is not None


def test_RSE_sort_order():
    tse = RangeSummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, RangeSummarizedExperiment)

    res = tse.order()
    assert res is not None
    assert len(res) == tse.shape[0]

    res = tse.sort(decreasing=True)
    assert res is not None
    assert res.rowRanges.shape == tse.rowRanges.shape
