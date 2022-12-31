import numpy as np
from random import random
import pandas as pd
import genomicranges
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

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

colData = pd.DataFrame({"treatment": ["ChIP", "Input"] * 3,})


def test_SE_props():
    tse = SummarizedExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assayNames is not None
    assert len(tse.assayNames) == 1

    assert tse.colData is not None
    assert tse.rowData is not None

    assert tse.dims == tse.shape

    assert tse.metadata is None


def test_SE_set_props():
    tse = SummarizedExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assayNames is not None

    tse.assayNames = ["normalized"]

    assert len(tse.assayNames) == 1

    tse.colData = None
    assert tse.colData is None

    tse.rowData = None
    assert tse.rowData is None

    assert tse.dims == tse.shape

    tse.metadata = {"something": "random"}
    assert tse.metadata is not None


def test_SE_subset():
    tse = SummarizedExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    subset_tse = tse[0:10, 2:5]
    assert subset_tse is not None
    assert isinstance(subset_tse, SummarizedExperiment)

    assert len(subset_tse.rowData) == 10
    assert len(subset_tse.colData) == 3

    assert subset_tse.assay("counts").shape == (10, 3)


def test_SE_subsetAssays():
    tse = SummarizedExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    subset_asys = tse.subsetAssays(rowIndices=slice(1, 10), colIndices=[0, 1, 2])
    assert subset_asys is not None
    assert isinstance(subset_asys, type(tse.assays))

    assert len(subset_asys.keys()) == 1
    assert subset_asys["counts"].shape == (9, 3)
