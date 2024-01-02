from random import random

import genomicranges
from biocframe import BiocFrame
import numpy as np
import pandas as pd
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


nrows = 200
ncols = 6
counts = np.random.rand(nrows, ncols)
row_data = BiocFrame(
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

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SE_init():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)
    assert tse.shape == (200, 6)
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)
    assert tse.row_names is None
    assert tse.col_names is not None


def test_SE_with_df():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data.to_pandas(), column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)
    assert tse.shape == (200, 6)
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)


def test_SE_no_row_or_col_data():
    tse = SummarizedExperiment(assays={"counts": counts})

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)
    assert tse.shape == (200, 6)
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)

    tse.row_names = [f"row_{i}" for i in range(200)]
    assert tse.rownames is not None
    assert len(tse.rownames) == 200
    assert tse.row_data.shape[0] == 200
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)

    tse.col_names = [f"col_{i}" for i in range(6)]
    assert tse.colnames is not None
    assert len(tse.colnames) == 6
    assert tse.col_data.shape[0] == 6
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)


def test_SE_export():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)
    assert tse.shape == (200, 6)
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)

    adata = tse.to_anndata()

    assert adata is not None
    assert adata.shape == (6, 200)
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)


def test_SE_no_assays():
    tse = SummarizedExperiment(
        row_data=BiocFrame(number_of_rows=10), column_data=BiocFrame(number_of_rows=3)
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)
    assert tse.shape == (10, 3)
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)
    assert tse.row_names is None
    assert tse.col_names is None


def test_SE_only_names():
    tse = SummarizedExperiment(
        row_names=["row_" + str(i) for i in range(10)],
        column_names=["col_" + str(i) for i in range(3)],
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)
    assert tse.shape == (10, 3)
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert len(tse.row_data) == 10
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)
    assert len(tse.col_data) == 3
    assert tse.row_names is not None
    assert tse.col_names is not None


def test_SE_empty():
    tse = SummarizedExperiment()

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)
    assert tse.shape == (0, 0)
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert len(tse.row_data) == 0
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)
    assert len(tse.col_data) == 0
    assert tse.row_names is None
    assert tse.col_names is None
