from random import random

import genomicranges
import numpy as np
import pandas as pd
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

gr = genomicranges.from_pandas(df_gr)

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SE_props():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assay_names is not None
    assert len(tse.assay_names) == 1

    assert tse.col_data is not None
    assert tse.row_data is not None

    assert tse.dims == tse.shape

    assert tse.metadata is None


def test_SE_set_props():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assay_names is not None

    tse.assay_names = ["normalized"]

    assert len(tse.assay_names) == 1

    tse.col_data = None
    assert tse.col_data is not None

    tse.row_data = None
    assert tse.row_data is not None

    assert tse.dims == tse.shape

    tse.metadata = {"something": "random"}
    assert tse.metadata is not None


def test_SE_subset_assays():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    subset_asys = tse.subset_assays(row_indices=slice(1, 10), col_indices=[0, 1, 2])
    assert subset_asys is not None
    assert isinstance(subset_asys, type(tse.assays))

    assert len(subset_asys.keys()) == 1
    assert subset_asys["counts"].shape == (9, 3)


def test_SE_assay():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assay_names is not None
    assert len(tse.assay_names) == 1

    assert tse.assay("counts") is not None
    assert tse.assay(0) is not None
