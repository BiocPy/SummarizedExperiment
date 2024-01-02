from random import random

import genomicranges
import numpy as np
import pandas as pd
import pytest
from summarizedexperiment.RangedSummarizedExperiment import RangedSummarizedExperiment

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

gr = genomicranges.GenomicRanges.from_pandas(df_gr)

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_RSE_props():
    tse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)

    assert tse.assay_names is not None
    assert len(tse.assay_names) == 1

    assert tse.col_data is not None
    assert tse.row_ranges is not None

    assert tse.dims == tse.shape

    assert tse.metadata is not None

    assert tse.start is not None
    assert tse.end is not None

    assert tse.seq_info is not None
    assert tse.seqnames is not None
    assert tse.strand is not None
    assert tse.width is not None

    assert tse.rownames is None
    assert tse.colnames is not None


def test_RSE_subset():
    tse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)

    subset_tse = tse[0:10, 2:5]
    assert subset_tse is not None
    assert isinstance(subset_tse, RangedSummarizedExperiment)

    assert len(subset_tse.row_ranges) == 10
    assert len(subset_tse.col_data) == 3

    assert subset_tse.assay("counts").shape == (10, 3)


def test_RSE_subset_assays():
    tse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)

    subset_asys = tse.subset_assays(rows=slice(1, 10), columns=[0, 1, 2])
    assert subset_asys is not None
    assert isinstance(subset_asys, type(tse.assays))

    assert len(subset_asys.keys()) == 1
    assert subset_asys["counts"].shape == (9, 3)


def test_RSE_coverage():
    tse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)

    cov = tse.coverage()
    assert cov is not None
    assert len(cov.keys()) == 3


def test_RSE_distance_methods():
    tse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)

    nearest = tse.nearest(tse)
    assert nearest is not None

    precede = tse.precede(tse)
    assert precede is not None

    follow = tse.follow(tse)
    assert follow is not None


def test_RSE_range_methods():
    tse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)

    res = tse.flank(width=10)
    assert res is not None

    res = tse.resize(width=15)
    assert res is not None

    res = tse.restrict(start=400)
    assert res is not None

    res = tse.shift(shift=25)
    assert res is not None

    res = tse.promoters()
    assert res is not None

    with pytest.raises(Exception):
        res = tse.narrow(start=40)


def test_RSE_search():
    tse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)

    res = tse.find_overlaps(tse)
    assert res is not None

    res = tse.subset_by_overlaps(tse)
    assert res is not None


def test_RSE_sort_order():
    tse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)

    res = tse.order()
    assert res is not None
    assert len(res) == tse.shape[0]

    res = tse.sort(decreasing=True)
    assert res is not None
    assert len(res.row_ranges) == len(tse.row_ranges)
