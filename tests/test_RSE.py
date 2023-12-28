from random import random

import genomicranges
from biocframe import BiocFrame
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

a = genomicranges.GenomicRanges.from_pandas(
    pd.DataFrame(
        {
            "seqnames": ["chr1", "chr2", "chr1", "chr3"],
            "starts": [1, 3, 2, 4],
            "ends": [10, 30, 50, 60],
            "strand": ["-", "+", "*", "+"],
            "score": [1, 2, 3, 4],
        }
    )
)

b = genomicranges.GenomicRanges.from_pandas(
    pd.DataFrame(
        {
            "seqnames": ["chr2", "chr4", "chr5"],
            "starts": [3, 6, 4],
            "ends": [30, 50, 60],
            "strand": ["-", "+", "*"],
            "score": [2, 3, 4],
        }
    )
)

grl = genomicranges.GenomicRangesList(ranges=[a, b], names=["a", "b"])


def test_RSE_creation():
    trse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, column_data=col_data
    )

    assert trse is not None
    assert isinstance(trse, RangedSummarizedExperiment)


def test_RSE_no_ranges():
    rse = RangedSummarizedExperiment(assays={"counts": counts})
    assert rse is not None
    assert isinstance(rse, RangedSummarizedExperiment)


def test_RSE_should_fail():
    with pytest.raises(Exception):
        RangedSummarizedExperiment(
            assays={"counts": counts}, row_data=BiocFrame(number_of_rows=10)
        )


def test_RSE_grl():
    counts = np.random.rand(2, ncols)

    trse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=grl, column_data=col_data
    )

    assert trse is not None
    assert isinstance(trse, RangedSummarizedExperiment)


def test_RSE_grl_should_fail():
    with pytest.raises(Exception):
        RangedSummarizedExperiment(assays={"counts": counts}, row_ranges=grl)


def test_RSE_empty():
    tse = RangedSummarizedExperiment()

    assert tse is not None
    assert isinstance(tse, RangedSummarizedExperiment)
    assert tse.shape == (0, 0)

    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert len(tse.row_data) == 0

    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)
    assert len(tse.col_data) == 0

    assert tse.row_ranges is not None
    assert isinstance(tse.row_ranges, genomicranges.GenomicRangesList)
    assert len(tse.row_ranges) == 0

    assert tse.row_names is None
    assert tse.col_names is None
