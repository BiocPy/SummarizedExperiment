from random import random

from biocframe import BiocFrame
import numpy as np
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from summarizedexperiment.RangedSummarizedExperiment import RangedSummarizedExperiment

import pytest

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

col_data = BiocFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SE_props():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assay_names is not None
    assert len(tse.assay_names) == 1

    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)

    assert tse.dims == tse.shape
    assert tse.metadata is not None


def test_SE_set_props():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assay_names is not None

    tse.assay_names = ["normalized"]

    assert len(tse.assay_names) == 1

    tse.row_data = None
    assert tse.row_data is not None
    assert isinstance(tse.row_data, BiocFrame)

    tse.col_data = None
    assert tse.col_data is not None
    assert isinstance(tse.col_data, BiocFrame)

    assert tse.dims == tse.shape

    tse.metadata = {"something": "random"}
    assert tse.metadata is not {}


def test_SE_assay():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assay_names is not None
    assert len(tse.assay_names) == 1

    assert tse.assay("counts") is not None
    assert tse.assay(0) is not None

def test_SE_assay_getters_and_setters():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    assert tse.assay(0) is not None

    new_tse = tse.set_assay("new_counts", assay=np.random.rand(nrows, ncols), in_place=False)
    assert new_tse.get_assay_names() != tse.get_assay_names()
    with pytest.raises(Exception):
        tse.get_assay("new_counts")
    assert new_tse.get_assay("new_counts") is not None

    tse.set_assay("new_counts", assay=np.random.rand(nrows, ncols), in_place=True)
    assert new_tse.get_assay_names() == tse.get_assay_names()
    assert tse.get_assay("new_counts") is not None
    assert new_tse.get_assay("new_counts") is not None

    mod_tse = tse.set_assay(0, assay=np.random.rand(nrows, ncols), in_place=False)
    assert mod_tse.get_assay_names() == tse.get_assay_names()
    assert tse.get_assay("new_counts") is not None
    assert mod_tse.get_assay("new_counts") is not None

    with pytest.raises(Exception):
        tse.set_assay(4, assay=np.random.rand(nrows, ncols), in_place=False)

    with pytest.raises(Exception):
        tse.set_assay(-1, assay=np.random.rand(nrows, ncols), in_place=False)

def test_SE_to_rse():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    rse = tse.to_rangedsummarizedexperiment()

    assert rse is not None
    assert isinstance(rse, RangedSummarizedExperiment)
    assert rse.shape == tse.shape

def test_SE_to_rse_parse_ranges():
    rd = BiocFrame({
        "seqnames": ["chr1"] * nrows,
        "starts": range(nrows),
        "ends": range(nrows),
        "strand": ["+"] * nrows
    })

    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=rd, column_data=col_data
    )

    rse = tse.to_rangedsummarizedexperiment()

    assert rse is not None
    assert isinstance(rse, RangedSummarizedExperiment)
    assert rse.shape == tse.shape

    assert rse.row_ranges is not None
    assert len(rse.row_ranges) == nrows
    assert np.allclose(rse.row_ranges.start, np.array(range(nrows)))

def test_SE_coldata_accessors():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse.get_column_data_column("treatment") is not None
    assert len(tse.get_column_data_column("treatment")) == 6

    new_tse = tse.set_column_data_column("stuff", [1, 2, 3, 4, 5, 6])
    assert new_tse.shape == tse.shape
    assert "stuff" in new_tse.col_data.column_names
    assert "stuff" not in tse.col_data.column_names

    tse.set_column_data_column("stuff", [1, 2, 3, 4, 5, 6], in_place=True)
    assert "stuff" in tse.col_data.column_names
