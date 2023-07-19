from random import random

import numpy as np
import pandas as pd
import pytest
from filebackedarray import H5BackedSparseData
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_SE_backed():
    # test_h5 = h5py.File("tests/data/tenx.sub.h5")

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
            * 100,
            "starts": range(0, 1000),
            "ends": range(0, 1000),
            "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * 100,
            "score": range(0, 1000),
            "GC": [random() for _ in range(10)] * 100,
        }
    )

    colData = pd.DataFrame(
        {
            "treatment": ["ChIP"] * 3005,
        }
    )

    assay = H5BackedSparseData("tests/data/tenx.sub.h5", "matrix")

    tse = SummarizedExperiment(
        assays={"counts_backed": assay},
        rowData=df_gr,
        colData=colData,
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)
    assert tse.shape == (1000, 3005)


def test_SE_backed_should_fail():
    # test_h5 = h5py.File("tests/data/tenx.sub.h5")

    with pytest.raises(Exception):
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

        colData = pd.DataFrame(
            {
                "treatment": ["ChIP", "Input"] * 3,
            }
        )

        assay = H5BackedSparseData("tests/data/tenx.sub.h5", "matrix")

        tse = SummarizedExperiment(
            assays={"counts_backed": assay, "counts": counts},
            rowData=df_gr,
            colData=colData,
        )

        assert tse is not None
        assert isinstance(tse, SummarizedExperiment)
        assert tse.shape == (200, 6)
