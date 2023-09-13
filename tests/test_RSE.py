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

gr = genomicranges.from_pandas(df_gr)

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)

a = genomicranges.GenomicRanges(
    {
        "seqnames": ["chr1", "chr2", "chr1", "chr3"],
        "starts": [1, 3, 2, 4],
        "ends": [10, 30, 50, 60],
        "strand": ["-", "+", "*", "+"],
        "score": [1, 2, 3, 4],
    }
)

b = genomicranges.GenomicRanges(
    {
        "seqnames": ["chr2", "chr4", "chr5"],
        "starts": [3, 6, 4],
        "ends": [30, 50, 60],
        "strand": ["-", "+", "*"],
        "score": [2, 3, 4],
    }
)

grl = genomicranges.GenomicRangesList(a=a, b=b)


def test_RSE_creation():
    trse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=gr, col_data=col_data
    )

    assert trse is not None
    assert isinstance(trse, RangedSummarizedExperiment)


def test_RSE_none_should_fail():
    with pytest.raises(Exception):
        RangedSummarizedExperiment(assays={"counts": counts})


def test_RSE_should_fail():
    with pytest.raises(Exception):
        RangedSummarizedExperiment(assays={"counts": counts}, row_ranges=df_gr)


def test_RSE_grl():
    counts = np.random.rand(2, ncols)

    trse = RangedSummarizedExperiment(
        assays={"counts": counts}, row_ranges=grl, col_data=col_data
    )

    assert trse is not None
    assert isinstance(trse, RangedSummarizedExperiment)


def test_RSE_grl_should_fail():
    with pytest.raises(Exception):
        RangedSummarizedExperiment(assays={"counts": counts}, row_ranges=grl)
