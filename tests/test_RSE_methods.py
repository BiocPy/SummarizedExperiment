from genomicranges import GenomicRanges
import numpy as np
from random import random
import pandas as pd

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

gr = GenomicRanges.fromPandas(df_gr)

colData = pd.DataFrame({"treatment": ["ChIP", "Input"] * 3,})


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
