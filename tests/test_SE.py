import pytest

from summarizedexperiment import SummarizedExperiment
from genomicranges import GenomicRanges
import numpy as np
from random import random
import pandas as pd
from summarizedexperiment.RangeSummarizedExperiment import (
    RangeSummarizedExperiment as rse,
)
from summarizedexperiment.SummarizedExperiment import (
    SummarizedExperiment as se,
)

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

gr = GenomicRanges.fromPandas(df_gr)

colData = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SE_creation():
    tse = SummarizedExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, se)


def test_RSE_creation():

    trse = SummarizedExperiment(
        assays={"counts": counts}, rowRanges=gr, colData=colData
    )

    assert trse is not None
    assert isinstance(trse, rse)


def test_SE_none():
    tse = SummarizedExperiment(assays={"counts": counts})

    assert tse is not None
    assert isinstance(tse, se)
